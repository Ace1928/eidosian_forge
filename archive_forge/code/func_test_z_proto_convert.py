import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_z_proto_convert():
    gate = cirq.Z ** sympy.Symbol('k')
    proto = operations_pb2.Operation(exp_z=operations_pb2.ExpZ(target=operations_pb2.Qubit(row=2, col=3), half_turns=operations_pb2.ParameterizedFloat(parameter_key='k')))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.Z ** 0.5
    proto = operations_pb2.Operation(exp_z=operations_pb2.ExpZ(target=operations_pb2.Qubit(row=2, col=3), half_turns=operations_pb2.ParameterizedFloat(raw=0.5)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))