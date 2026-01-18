import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_measurement_gate_multiple_keys():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result == ionq.SerializedProgram(body={'gateset': 'native', 'qubits': 2, 'circuit': []}, metadata={'measurement0': f'a{chr(31)}0{chr(30)}b{chr(31)}1'}, settings={})