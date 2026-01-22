from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _serialize_circuit(self, circuit: cirq.AbstractCircuit, msg: v2.program_pb2.Circuit, *, arg_function_language: Optional[str], constants: List[v2.program_pb2.Constant], raw_constants: Dict[Any, int]) -> None:
    msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
    for moment in circuit:
        moment_proto = msg.moments.add()
        for op in moment:
            if isinstance(op.untagged, cirq.CircuitOperation):
                op_pb = moment_proto.circuit_operations.add()
                self._serialize_circuit_op(op.untagged, op_pb, arg_function_language=arg_function_language, constants=constants, raw_constants=raw_constants)
            else:
                op_pb = moment_proto.operations.add()
                self._serialize_gate_op(op, op_pb, arg_function_language=arg_function_language, constants=constants, raw_constants=raw_constants)