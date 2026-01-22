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