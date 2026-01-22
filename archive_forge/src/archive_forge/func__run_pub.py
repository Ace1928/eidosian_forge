from __future__ import annotations
from collections.abc import Iterable
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from .base import BaseEstimatorV2
from .containers import EstimatorPubLike, PrimitiveResult, PubResult
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob
from .utils import bound_circuit_to_instruction
def _run_pub(self, pub: EstimatorPub) -> PubResult:
    rng = np.random.default_rng(self._seed)
    circuit = pub.circuit
    observables = pub.observables
    parameter_values = pub.parameter_values
    precision = pub.precision
    bound_circuits = parameter_values.bind_all(circuit)
    bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
    evs = np.zeros_like(bc_circuits, dtype=np.float64)
    stds = np.zeros_like(bc_circuits, dtype=np.float64)
    for index in np.ndindex(*bc_circuits.shape):
        bound_circuit = bc_circuits[index]
        observable = bc_obs[index]
        final_state = Statevector(bound_circuit_to_instruction(bound_circuit))
        paulis, coeffs = zip(*observable.items())
        obs = SparsePauliOp(paulis, coeffs)
        expectation_value = np.real_if_close(final_state.expectation_value(obs))
        if precision != 0:
            if not np.isreal(expectation_value):
                raise ValueError('Given operator is not Hermitian and noise cannot be added.')
            expectation_value = rng.normal(expectation_value, precision)
        evs[index] = expectation_value
    data_bin_cls = self._make_data_bin(pub)
    data_bin = data_bin_cls(evs=evs, stds=stds)
    return PubResult(data_bin, metadata={'precision': precision})