import numpy as np
import pytest
import sympy
import cirq
import cirq_ionq as ionq
def test_serialize_empty_circuit_invalid():
    empty = cirq.Circuit()
    serializer = ionq.Serializer()
    with pytest.raises(ValueError, match='empty'):
        _ = serializer.serialize(empty)