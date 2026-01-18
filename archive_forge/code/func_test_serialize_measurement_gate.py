import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_measurement_gate():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='tomyheart'))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'native', 'qubits': 1, 'circuit': []}, metadata={'measurement0': f'tomyheart{chr(31)}0'}, settings={})