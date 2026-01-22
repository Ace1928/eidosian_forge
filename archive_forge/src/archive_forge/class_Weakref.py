import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class Weakref(trace.TraceType):
    """Represents weakref of an arbitrary Python object.

  When a function argument is a custom class, instead of making a copy of it
  just for the sake of function cache, a weakref is instead kept to save memory.
  """

    def __init__(self, ref: weakref.ReferenceType):
        self._ref = ref
        self._ref_hash = hash(ref)

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        return self == other

    def most_specific_common_supertype(self, types: Sequence[trace.TraceType]) -> Optional['Weakref']:
        return self if all((self == other for other in types)) else None

    def placeholder_value(self, placeholder_context) -> Any:
        return self._ref()

    def _cast(self, value, _):
        if value is self._ref() or value == self._ref():
            return value
        while hasattr(value, '__wrapped__'):
            value = value.__wrapped__
            if value is self._ref():
                return value
        raise ValueError(f'Can not cast {value!r} to {self!r}')

    def __eq__(self, other):
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, Weakref):
            return False
        if self._ref() is None or other._ref() is None:
            return False
        if self._ref() is other._ref():
            return True
        return self._ref == other._ref

    def __hash__(self):
        return self._ref_hash

    def __repr__(self):
        return f'{self.__class__.__name__}(ref={self._ref!r})'