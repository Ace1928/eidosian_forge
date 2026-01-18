import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_s_gate():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.Z(q0) ** 0.5)
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 's', 'targets': [0]}]}, metadata={}, settings={})
    circuit = cirq.Circuit(cirq.Z(q0) ** (-0.5))
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'qis', 'qubits': 1, 'circuit': [{'gate': 'si', 'targets': [0]}]}, metadata={}, settings={})