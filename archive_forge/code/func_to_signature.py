import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def to_signature(x_type):
    if x_type is None:
        raise TypeError(f'Can not generate structured signature if FunctionType is not fully specified. Received {function_type}')
    return x_type.placeholder_value(trace_type.InternalPlaceholderContext(unnest_only=True))