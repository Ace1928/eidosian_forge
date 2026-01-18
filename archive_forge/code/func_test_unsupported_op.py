import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_unsupported_op():
    with pytest.raises(ValueError, match='invalid operation'):
        programs.xmon_op_from_proto(operations_pb2.Operation())
    with pytest.raises(ValueError, match='know how to serialize'):
        programs.gate_to_proto(cirq.CCZ, (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)), delay=0)