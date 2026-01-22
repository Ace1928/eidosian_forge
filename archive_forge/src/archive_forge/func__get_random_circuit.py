import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def _get_random_circuit(qubits, *, n_moments=10, random_state=52):
    return cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(qubits=qubits, depth=n_moments, seed=random_state, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)) + cirq.measure(*qubits, key='z')