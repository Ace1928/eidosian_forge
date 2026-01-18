import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_non_gate_op_invalid():
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.CircuitOperation(cirq.FrozenCircuit()))
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='CircuitOperation'):
        _ = serializer.serialize(circuit)