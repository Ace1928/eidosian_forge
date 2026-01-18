import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_measurement_key_cannot_be_deliminator():
    q0 = cirq.LineQubit(0)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(30)}'))
    with pytest.raises(ValueError, match=f'ab{chr(30)}'):
        _ = serializer.serialize(circuit)
    circuit = cirq.Circuit(cirq.measure(q0, key=f'ab{chr(31)}'))
    with pytest.raises(ValueError, match=f'ab{chr(31)}'):
        _ = serializer.serialize(circuit)