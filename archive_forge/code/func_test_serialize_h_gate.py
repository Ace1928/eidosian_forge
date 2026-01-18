import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_h_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.H(q0))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'h', 'targets': [0]}]}, metadata={}, settings={})
    with pytest.raises(ValueError, match='H\\*\\*0.5'):
        circuit = cirq.Circuit(cirq.H(q0) ** 0.5)
        _ = serializer.serialize(circuit)