from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import warnings
import numpy as np
from numpy.typing import NDArray
from qiskit import ClassicalRegister, QiskitError, QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.quantum_info import Statevector
from .base import BaseSamplerV2
from .base.validation import _has_measure
from .containers import (
from .containers.sampler_pub import SamplerPub
from .containers.bit_array import _min_num_bytes
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction
class StatevectorSampler(BaseSamplerV2):
    """
    Simple implementation of :class:`BaseSamplerV2` using full state vector simulation.

    This class is implemented via :class:`~.Statevector` which turns provided circuits into
    pure state vectors, and is therefore incompatible with mid-circuit measurements (although
    other implementations may be).

    As seen in the example below, this sampler supports providing arrays of parameter value sets to
    bind against a single circuit.

    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    .. code-block:: python

        from qiskit.circuit import (
            Parameter, QuantumCircuit, ClassicalRegister, QuantumRegister
        )
        from qiskit.primitives import StatevectorSampler

        import matplotlib.pyplot as plt
        import numpy as np

        # Define our circuit registers, including classical registers
        # called 'alpha' and 'beta'.
        qreg = QuantumRegister(3)
        alpha = ClassicalRegister(2, "alpha")
        beta = ClassicalRegister(1, "beta")

        # Define a quantum circuit with two parameters.
        circuit = QuantumCircuit(qreg, alpha, beta)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.ry(Parameter("a"), 0)
        circuit.rz(Parameter("b"), 0)
        circuit.cx(1, 2)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.measure([0, 1], alpha)
        circuit.measure([2], beta)

        # Define a sweep over parameter values, where the second axis is over.
        # the two parameters in the circuit.
        params = np.vstack([
            np.linspace(-np.pi, np.pi, 100),
            np.linspace(-4 * np.pi, 4 * np.pi, 100)
        ]).T

        # Instantiate a new statevector simulation based sampler object.
        sampler = StatevectorSampler()

        # Start a job that will return shots for all 100 parameter value sets.
        pub = (circuit, params)
        job = sampler.run([pub], shots=256)

        # Extract the result for the 0th pub (this example only has one pub).
        result = job.result()[0]

        # There is one BitArray object for each ClassicalRegister in the
        # circuit. Here, we can see that the BitArray for alpha contains data
        # for all 100 sweep points, and that it is indeed storing data for 2
        # bits over 256 shots.
        assert result.data.alpha.shape == (100,)
        assert result.data.alpha.num_bits == 2
        assert result.data.alpha.num_shots == 256

        # We can work directly with a binary array in performant applications.
        raw = result.data.alpha.array

        # For small registers where it is anticipated to have many counts
        # associated with the same bitstrings, we can turn the data from,
        # for example, the 22nd sweep index into a dictionary of counts.
        counts = result.data.alpha.get_counts(22)

        # Or, convert into a list of bitstrings that preserve shot order.
        bitstrings = result.data.alpha.get_bitstrings(22)
        print(bitstrings)

    """

    def __init__(self, *, default_shots: int=1024, seed: np.random.Generator | int | None=None):
        """
        Args:
            default_shots: The default shots for the sampler if not specified during run.
            seed: The seed or Generator object for random number generation.
                If None, a random seeded default RNG will be used.
        """
        self._default_shots = default_shots
        self._seed = seed

    @property
    def default_shots(self) -> int:
        """Return the default shots"""
        return self._default_shots

    @property
    def seed(self) -> np.random.Generator | int | None:
        """Return the seed or Generator object for random number generation."""
        return self._seed

    def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None=None) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        if any((len(pub.circuit.cregs) == 0 for pub in coerced_pubs)):
            warnings.warn('One of your circuits has no output classical registers and so the result will be empty. Did you mean to add measurement instructions?', UserWarning)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[PubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: SamplerPub) -> PubResult:
        circuit, qargs, meas_info = _preprocess_circuit(pub.circuit)
        bound_circuits = pub.parameter_values.bind_all(circuit)
        arrays = {item.creg_name: np.zeros(bound_circuits.shape + (pub.shots, item.num_bytes), dtype=np.uint8) for item in meas_info}
        for index, bound_circuit in np.ndenumerate(bound_circuits):
            final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
            final_state.seed(self._seed)
            if qargs:
                samples = final_state.sample_memory(shots=pub.shots, qargs=qargs)
            else:
                samples = [''] * pub.shots
            samples_array = np.array([np.fromiter(sample, dtype=np.uint8) for sample in samples])
            for item in meas_info:
                ary = _samples_to_packed_array(samples_array, item.num_bits, item.qreg_indices)
                arrays[item.creg_name][index] = ary
        data_bin_cls = make_data_bin([(item.creg_name, BitArray) for item in meas_info], shape=bound_circuits.shape)
        meas = {item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info}
        data_bin = data_bin_cls(**meas)
        return PubResult(data_bin, metadata={'shots': pub.shots})