import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
import pytest
import numpy as np
from scipy.stats import unitary_group
@pytest.mark.parametrize('n_qubits', list(range(1, 8)))
def test_random_qsd_n_qubit(n_qubits):
    U = unitary_group.rvs(2 ** n_qubits)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(n_qubits)]
    circuit = cirq.Circuit(quantum_shannon_decomposition(qubits, U))
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-09)
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all((isinstance(op.gate, gates) for op in circuit.all_operations()))