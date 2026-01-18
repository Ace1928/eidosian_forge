from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
def test_serialize_deserialize_circuit_with_tokens():
    serializer = cg.CircuitSerializer()
    tag1 = cg.CalibrationTag('abc123')
    tag2 = cg.CalibrationTag('def456')
    circuit = cirq.Circuit(cirq.X(Q0).with_tags(tag1), cirq.X(Q1).with_tags(tag2), cirq.X(Q0).with_tags(tag2), cirq.X(Q0))
    op_q0_tag1 = v2.program_pb2.Operation()
    op_q0_tag1.xpowgate.exponent.float_value = 1.0
    op_q0_tag1.qubit_constant_index.append(0)
    op_q0_tag1.token_constant_index = 1
    op_q1_tag2 = v2.program_pb2.Operation()
    op_q1_tag2.xpowgate.exponent.float_value = 1.0
    op_q1_tag2.qubit_constant_index.append(2)
    op_q1_tag2.token_constant_index = 3
    op_q0_tag2 = v2.program_pb2.Operation()
    op_q0_tag2.xpowgate.exponent.float_value = 1.0
    op_q0_tag2.qubit_constant_index.append(0)
    op_q0_tag2.token_constant_index = 3
    proto = v2.program_pb2.Program(language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME), circuit=v2.program_pb2.Circuit(scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[v2.program_pb2.Moment(operations=[op_q0_tag1, op_q1_tag2]), v2.program_pb2.Moment(operations=[op_q0_tag2]), v2.program_pb2.Moment(operations=[X_PROTO])]), constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4')), v2.program_pb2.Constant(string_value='abc123'), v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_5')), v2.program_pb2.Constant(string_value='def456')])
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit