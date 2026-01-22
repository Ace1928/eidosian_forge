import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def _make_validated_mono_param(name, value, kind, type_context, poly_type) -> Parameter:
    """Generates and validates a parameter for Monomorphic FunctionType."""
    mono_type = trace_type.from_value(value, type_context)
    if poly_type and (not mono_type.is_subtype_of(poly_type)):
        raise TypeError(f'Parameter `{name}` was expected to be of type {poly_type} but is {mono_type}')
    return Parameter(name, kind, False, mono_type)