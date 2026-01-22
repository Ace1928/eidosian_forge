from __future__ import annotations
from collections.abc import Sequence
from itertools import accumulate
import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV1, BackendV2, Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import Counts, Result
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
from .base import BaseEstimator, EstimatorResult
from .primitive_job import PrimitiveJob
from .utils import _circuit_key, _observable_key, init_observable
class BackendEstimator(BaseEstimator[PrimitiveJob[EstimatorResult]]):
    """Evaluates expectation value using Pauli rotation gates.

    The :class:`~.BackendEstimator` class is a generic implementation of the
    :class:`~.BaseEstimator` interface that is used to wrap a :class:`~.BackendV2`
    (or :class:`~.BackendV1`) object in the :class:`~.BaseEstimator` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseEstimator` implementation in places that work with
    :class:`~.BaseEstimator`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseEstimator`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.
    """

    def __init__(self, backend: BackendV1 | BackendV2, options: dict | None=None, abelian_grouping: bool=True, bound_pass_manager: PassManager | None=None, skip_transpilation: bool=False):
        """Initialize a new BackendEstimator instance

        Args:
            backend: Required: the backend to run the primitive on
            options: Default options.
            abelian_grouping: Whether the observable should be grouped into
                commuting
            bound_pass_manager: An optional pass manager to run after
                parameter binding.
            skip_transpilation: If this is set to True the internal compilation
                of the input circuits is skipped and the circuit objects
                will be directly executed when this object is called.
        """
        super().__init__(options=options)
        self._circuits = []
        self._parameters = []
        self._observables = []
        self._abelian_grouping = abelian_grouping
        self._backend = backend
        self._transpile_options = Options()
        self._bound_pass_manager = bound_pass_manager
        self._preprocessed_circuits: list[tuple[QuantumCircuit, list[QuantumCircuit]]] | None = None
        self._transpiled_circuits: list[QuantumCircuit] | None = None
        self._grouping = list(zip(range(len(self._circuits)), range(len(self._observables))))
        self._skip_transpilation = skip_transpilation
        self._circuit_ids = {}
        self._observable_ids = {}

    @property
    def transpile_options(self) -> Options:
        """Return the transpiler options for transpiling the circuits."""
        return self._transpile_options

    def set_transpile_options(self, **fields):
        """Set the transpiler options for transpiler.
        Args:
            **fields: The fields to update the options
        """
        self._transpiled_circuits = None
        self._transpile_options.update_options(**fields)

    @property
    def preprocessed_circuits(self) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Transpiled quantum circuits produced by preprocessing
        Returns:
            List of the transpiled quantum circuit
        """
        self._preprocessed_circuits = self._preprocessing()
        return self._preprocessed_circuits

    @property
    def transpiled_circuits(self) -> list[QuantumCircuit]:
        """
        Transpiled quantum circuits.
        Returns:
            List of the transpiled quantum circuit
        Raises:
            QiskitError: if the instance has been closed.
        """
        self._transpile()
        return self._transpiled_circuits

    @property
    def backend(self) -> BackendV1 | BackendV2:
        """
        Returns:
            The backend which this estimator object based on
        """
        return self._backend

    def _transpile(self):
        """Split Transpile"""
        self._transpiled_circuits = []
        for common_circuit, diff_circuits in self.preprocessed_circuits:
            if self._skip_transpilation:
                transpiled_circuit = common_circuit.copy()
                final_index_layout = list(range(common_circuit.num_qubits))
            else:
                transpiled_circuit = transpile(common_circuit, self.backend, **self.transpile_options.__dict__)
                if transpiled_circuit.layout is not None:
                    final_index_layout = transpiled_circuit.layout.final_index_layout()
                else:
                    final_index_layout = list(range(transpiled_circuit.num_qubits))
            passmanager = _passmanager_for_measurement_circuits(final_index_layout, self.backend)
            diff_circuits = passmanager.run(diff_circuits)
            transpiled_circuits = []
            for diff_circuit in diff_circuits:
                transpiled_circuit_copy = transpiled_circuit.copy()
                clbits = diff_circuit.cregs[0]
                for creg in transpiled_circuit_copy.cregs:
                    if clbits.name == creg.name:
                        raise QiskitError(f"Classical register for measurements conflict with those of the input circuit: {clbits}. Recommended to avoid register names starting with '__'.")
                transpiled_circuit_copy.add_register(clbits)
                transpiled_circuit_copy.compose(diff_circuit, clbits=clbits, inplace=True)
                transpiled_circuit_copy.metadata = diff_circuit.metadata
                transpiled_circuits.append(transpiled_circuit_copy)
            self._transpiled_circuits += transpiled_circuits

    def _call(self, circuits: Sequence[int], observables: Sequence[int], parameter_values: Sequence[Sequence[float]], **run_options) -> EstimatorResult:
        self._grouping = list(zip(circuits, observables))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for _, m in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))
        parameter_dicts = [dict(zip(self._parameters[i], value)) for i, value in zip(circuits, parameter_values)]
        bound_circuits = [transpiled_circuits[circuit_index] if len(p) == 0 else transpiled_circuits[circuit_index].assign_parameters(p) for i, (p, n) in enumerate(zip(parameter_dicts, num_observables)) for circuit_index in range(accum[i], accum[i] + n)]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)
        result, metadata = _run_circuits(bound_circuits, self._backend, **run_options)
        return self._postprocessing(result, accum, metadata)

    def _run(self, circuits: tuple[QuantumCircuit, ...], observables: tuple[BaseOperator, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options):
        circuit_indices = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(self._observables)
                self._observables.append(observable)
        job = PrimitiveJob(self._call, circuit_indices, observable_indices, parameter_values, **run_options)
        job._submit()
        return job

    @staticmethod
    def _measurement_circuit(num_qubits: int, pauli: Pauli):
        qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
        if not np.any(qubit_indices):
            qubit_indices = [0]
        meas_circuit = QuantumCircuit(QuantumRegister(num_qubits, 'q'), ClassicalRegister(len(qubit_indices), f'__c_{pauli}'))
        for clbit, i in enumerate(qubit_indices):
            if pauli.x[i]:
                if pauli.z[i]:
                    meas_circuit.sdg(i)
                meas_circuit.h(i)
            meas_circuit.measure(i, clbit)
        return (meas_circuit, qubit_indices)

    def _preprocessing(self) -> list[tuple[QuantumCircuit, list[QuantumCircuit]]]:
        """
        Preprocessing for evaluation of expectation value using pauli rotation gates.
        """
        preprocessed_circuits = []
        for group in self._grouping:
            circuit = self._circuits[group[0]]
            observable = self._observables[group[1]]
            diff_circuits: list[QuantumCircuit] = []
            if self._abelian_grouping:
                for obs in observable.group_commuting(qubit_wise=True):
                    basis = Pauli((np.logical_or.reduce(obs.paulis.z), np.logical_or.reduce(obs.paulis.x)))
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(obs.paulis.z[:, indices], obs.paulis.x[:, indices], obs.paulis.phase)
                    meas_circuit.metadata = {'paulis': paulis, 'coeffs': np.real_if_close(obs.coeffs)}
                    diff_circuits.append(meas_circuit)
            else:
                for basis, obs in zip(observable.paulis, observable):
                    meas_circuit, indices = self._measurement_circuit(circuit.num_qubits, basis)
                    paulis = PauliList.from_symplectic(obs.paulis.z[:, indices], obs.paulis.x[:, indices], obs.paulis.phase)
                    meas_circuit.metadata = {'paulis': paulis, 'coeffs': np.real_if_close(obs.coeffs)}
                    diff_circuits.append(meas_circuit)
            preprocessed_circuits.append((circuit.copy(), diff_circuits))
        return preprocessed_circuits

    def _postprocessing(self, result: list[Result], accum: list[int], metadata: list[dict]) -> EstimatorResult:
        """
        Postprocessing for evaluation of expectation value using pauli rotation gates.
        """
        counts = _prepare_counts(result)
        expval_list = []
        var_list = []
        shots_list = []
        for i, j in zip(accum, accum[1:]):
            combined_expval = 0.0
            combined_var = 0.0
            for k in range(i, j):
                meta = metadata[k]
                paulis = meta['paulis']
                coeffs = meta['coeffs']
                count = counts[k]
                expvals, variances = _pauli_expval_with_variance(count, paulis)
                combined_expval += np.dot(expvals, coeffs)
                combined_var += np.dot(variances, coeffs ** 2)
            expval_list.append(combined_expval)
            var_list.append(combined_var)
            shots_list.append(sum(counts[i].values()))
        metadata = [{'variance': var, 'shots': shots} for var, shots in zip(var_list, shots_list)]
        return EstimatorResult(np.real_if_close(expval_list), metadata)

    def _bound_pass_manager_run(self, circuits):
        if self._bound_pass_manager is None:
            return circuits
        else:
            output = self._bound_pass_manager.run(circuits)
            if not isinstance(output, list):
                output = [output]
            return output