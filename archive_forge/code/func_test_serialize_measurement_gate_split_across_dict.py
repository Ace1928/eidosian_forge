import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_measurement_gate_split_across_dict():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='a' * 60))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result.metadata['measurement0'] == 'a' * 40
    assert result.metadata['measurement1'] == 'a' * 20 + f'{chr(31)}0'