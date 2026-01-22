from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import numbers
import abc
import numpy as np
import cirq
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto
class CircuitOpSerializer(OpSerializer):
    """Describes how to serialize CircuitOperations."""

    @property
    def internal_type(self):
        return cirq.FrozenCircuit

    @property
    def serialized_id(self):
        return 'circuit'

    @property
    def can_serialize_predicate(self):
        return lambda op: isinstance(op.untagged, cirq.CircuitOperation)

    def to_proto(self, op: cirq.CircuitOperation, msg: Optional[v2.program_pb2.CircuitOperation]=None, *, arg_function_language: Optional[str]='', constants: List[v2.program_pb2.Constant], raw_constants: Dict[Any, int]) -> v2.program_pb2.CircuitOperation:
        """Returns the cirq.google.api.v2.CircuitOperation message as a proto dict.

        Note that this function requires constants and raw_constants to be
        pre-populated with the circuit in op.
        """
        if not isinstance(op, cirq.CircuitOperation):
            raise ValueError(f'Serializer expected CircuitOperation but got {type(op)}.')
        msg = msg or v2.program_pb2.CircuitOperation()
        try:
            msg.circuit_constant_index = raw_constants[op.circuit]
        except KeyError as err:
            raise ValueError(f'Encountered a circuit not in the constants table. Full error message:\n{err}')
        if op.repetition_ids is not None and op.repetition_ids != circuit_operation.default_repetition_ids(op.repetitions):
            for rep_id in op.repetition_ids:
                msg.repetition_specification.repetition_ids.ids.append(rep_id)
        elif isinstance(op.repetitions, (int, np.integer)):
            msg.repetition_specification.repetition_count = int(op.repetitions)
        else:
            raise ValueError(f'Cannot serialize repetitions of type {type(op.repetitions)}')
        for q1, q2 in op.qubit_map.items():
            entry = msg.qubit_map.entries.add()
            entry.key.id = v2.qubit_to_proto_id(q1)
            entry.value.id = v2.qubit_to_proto_id(q2)
        for mk1, mk2 in op.measurement_key_map.items():
            entry = msg.measurement_key_map.entries.add()
            entry.key.string_key = mk1
            entry.value.string_key = mk2
        for p1, p2 in op.param_resolver.param_dict.items():
            entry = msg.arg_map.entries.add()
            arg_to_proto(p1, out=entry.key, arg_function_language=arg_function_language)
            if isinstance(p2, (complex, numbers.Complex)):
                if isinstance(p2, numbers.Real):
                    p2 = float(p2)
                else:
                    raise ValueError(f'Cannot serialize complex value {p2}')
            arg_to_proto(p2, out=entry.value, arg_function_language=arg_function_language)
        return msg