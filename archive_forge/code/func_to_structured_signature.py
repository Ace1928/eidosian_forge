import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def to_structured_signature(function_type: FunctionType) -> Tuple[Any, Any]:
    """Returns structured input and output signatures from a FunctionType."""

    def to_signature(x_type):
        if x_type is None:
            raise TypeError(f'Can not generate structured signature if FunctionType is not fully specified. Received {function_type}')
        return x_type.placeholder_value(trace_type.InternalPlaceholderContext(unnest_only=True))
    args_signature = []
    kwargs_signature = {}
    for p in function_type.parameters.values():
        if p.kind == Parameter.POSITIONAL_ONLY:
            args_signature.append(to_signature(p.type_constraint))
        else:
            kwargs_signature[p.name] = to_signature(p.type_constraint)
    input_signature = (tuple(args_signature), kwargs_signature)
    output_signature = to_signature(function_type.output)
    return (input_signature, output_signature)