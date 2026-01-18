import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_not_serializable():
    q0, q1 = cirq.LineQubit.range(2)
    serializer = ionq.Serializer()
    circuit = cirq.Circuit(cirq.PhasedISwapPowGate()(q0, q1))
    with pytest.raises(ValueError, match='PhasedISWAP'):
        _ = serializer.serialize(circuit)