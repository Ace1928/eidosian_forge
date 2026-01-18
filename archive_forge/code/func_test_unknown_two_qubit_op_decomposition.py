import cirq
import pytest
import numpy as np
import cirq_google as cg
import sympy
@pytest.mark.parametrize('op', [cirq.CircuitOperation(cirq.FrozenCircuit(cirq.SWAP(*_QUBITS), cirq.ZZ(*_QUBITS), cirq.SWAP(*_QUBITS))), cirq.X(_QUBITS[0]), cirq.XX(*_QUBITS) ** _THETA, cirq.FSimGate(0.25, 0.85).on(*_QUBITS), cirq.XX(*_QUBITS), cirq.YY(*_QUBITS), *[cirq.testing.random_unitary(4, random_state=1234) for _ in range(10)]])
def test_unknown_two_qubit_op_decomposition(op):
    assert cg.known_2q_op_to_sycamore_operations(op) is None
    if cirq.has_unitary(op) and cirq.num_qubits(op) == 2:
        matrix_2q_circuit = cirq.Circuit(cg.two_qubit_matrix_to_sycamore_operations(_QUBITS[0], _QUBITS[1], cirq.unitary(op)))
        assert_implements(matrix_2q_circuit, op)