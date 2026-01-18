import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_non_terminal_measurements():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q0, key='d'), cirq.X(q0))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='end of circuit'):
        _ = serializer.serialize(circuit)