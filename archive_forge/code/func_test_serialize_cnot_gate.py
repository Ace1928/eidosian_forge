import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_cnot_gate():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.CNOT(q0, q1))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 2, 'circuit': [{'gate': 'cnot', 'control': 0, 'target': 1}]}, metadata={}, settings={})
    with pytest.raises(ValueError, match='CNOT\\*\\*0.5'):
        circuit = cirq.Circuit(cirq.CNOT(q0, q1) ** 0.5)
        _ = serializer.serialize(circuit)