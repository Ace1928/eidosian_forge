import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
def test_random_single_qubit_decomposition():
    U = unitary_group.rvs(2)
    qubit = cirq.NamedQubit('q0')
    circuit = cirq.Circuit(_single_qubit_decomposition(qubit, U))
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-09)
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all((isinstance(op.gate, gates) for op in circuit.all_operations()))