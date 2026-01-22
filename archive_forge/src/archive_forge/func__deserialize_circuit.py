from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _deserialize_circuit(self, circuit_proto: v2.program_pb2.Circuit, *, arg_function_language: str, constants: List[v2.program_pb2.Constant], deserialized_constants: List[Any]) -> cirq.Circuit:
    moments = []
    for moment_proto in circuit_proto.moments:
        moment_ops = []
        for op in moment_proto.operations:
            moment_ops.append(self._deserialize_gate_op(op, arg_function_language=arg_function_language, constants=constants, deserialized_constants=deserialized_constants))
        for op in moment_proto.circuit_operations:
            moment_ops.append(self._deserialize_circuit_op(op, arg_function_language=arg_function_language, constants=constants, deserialized_constants=deserialized_constants))
        moments.append(cirq.Moment(moment_ops))
    return cirq.Circuit(moments)