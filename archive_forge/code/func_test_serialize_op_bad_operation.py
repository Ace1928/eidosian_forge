from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_serialize_op_bad_operation():
    serializer = cg.CircuitSerializer()

    class NullOperation(cirq.Operation):

        @property
        def qubits(self):
            return tuple()

        def with_qubits(self, *qubits):
            return self
    null_op = NullOperation()
    with pytest.raises(ValueError, match='Cannot serialize op'):
        serializer.serialize(cirq.Circuit(null_op))