import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_w_to_proto():
    gate = cirq.PhasedXPowGate(exponent=sympy.Symbol('k'), phase_exponent=1)
    proto = operations_pb2.Operation(exp_w=operations_pb2.ExpW(target=operations_pb2.Qubit(row=2, col=3), axis_half_turns=operations_pb2.ParameterizedFloat(raw=1), half_turns=operations_pb2.ParameterizedFloat(parameter_key='k')))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=sympy.Symbol('j'))
    proto = operations_pb2.Operation(exp_w=operations_pb2.ExpW(target=operations_pb2.Qubit(row=2, col=3), axis_half_turns=operations_pb2.ParameterizedFloat(parameter_key='j'), half_turns=operations_pb2.ParameterizedFloat(raw=0.5)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.X ** 0.25
    proto = operations_pb2.Operation(exp_w=operations_pb2.ExpW(target=operations_pb2.Qubit(row=2, col=3), axis_half_turns=operations_pb2.ParameterizedFloat(raw=0.0), half_turns=operations_pb2.ParameterizedFloat(raw=0.25)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.Y ** 0.25
    proto = operations_pb2.Operation(exp_w=operations_pb2.ExpW(target=operations_pb2.Qubit(row=2, col=3), axis_half_turns=operations_pb2.ParameterizedFloat(raw=0.5), half_turns=operations_pb2.ParameterizedFloat(raw=0.25)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))
    gate = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=sympy.Symbol('j'))
    proto = operations_pb2.Operation(exp_w=operations_pb2.ExpW(target=operations_pb2.Qubit(row=2, col=3), axis_half_turns=operations_pb2.ParameterizedFloat(parameter_key='j'), half_turns=operations_pb2.ParameterizedFloat(raw=0.5)))
    assert_proto_dict_convert(gate, proto, cirq.GridQubit(2, 3))