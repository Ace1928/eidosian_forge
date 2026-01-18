from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_serialize_op_unsupported_type():
    serializer = cg.CircuitSerializer()
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='CNOT'):
        serializer.serialize(cirq.Circuit(cirq.CNOT(q0, q1)))