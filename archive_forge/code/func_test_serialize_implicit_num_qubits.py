import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_implicit_num_qubits():
    q0 = cirq.LineQubit(2)
    circuit = cirq.Circuit(cirq.X(q0))
    serializer = ionq.Serializer()
    result = serializer.serialize(circuit)
    assert result.body['qubits'] == 3