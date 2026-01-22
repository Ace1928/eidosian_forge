import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class Tuple(trace.TraceType, serialization.Serializable):
    """Represents a tuple of TraceType objects."""

    def __init__(self, *components: trace.TraceType):
        self.components = components

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        if not isinstance(other, Tuple) or len(self.components) != len(other.components):
            return False
        return all((self_component.is_subtype_of(other_component) for self_component, other_component in zip(self.components, other.components)))

    def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional['Tuple']:
        """See base class."""
        if not all((isinstance(other, Tuple) and len(self.components) == len(other.components) for other in others)):
            return None
        supertyped_components = []
        for i, component in enumerate(self.components):
            supertyped_component = component.most_specific_common_supertype([other.components[i] for other in others])
            if supertyped_component is None:
                return None
            supertyped_components.append(supertyped_component)
        return Tuple(*supertyped_components)

    @classmethod
    def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedTuple]:
        return default_types_pb2.SerializedTuple

    @classmethod
    def experimental_from_proto(cls, proto: default_types_pb2.SerializedTuple) -> 'Tuple':
        return Tuple(*[serialization.deserialize(c) for c in proto.components])

    def experimental_as_proto(self) -> default_types_pb2.SerializedTuple:
        return default_types_pb2.SerializedTuple(components=[serialization.serialize(c) for c in self.components])

    def placeholder_value(self, placeholder_context) -> Any:
        components = [component.placeholder_value(placeholder_context) for component in self.components]
        return tuple(components)

    def _to_tensors(self, value) -> Any:
        assert isinstance(value, tuple)
        flattened_values = []
        for comp_value, comp_type in zip(value, self.components):
            flattened_values.extend(comp_type._to_tensors(comp_value))
        return flattened_values

    def _from_tensors(self, tensors) -> Any:
        return tuple((c._from_tensors(tensors) for c in self.components))

    def _flatten(self) -> PythonList[trace.TraceType]:
        flattened_types = []
        for component in self.components:
            flattened_types.extend(component._flatten())
        return flattened_types

    def _cast(self, value: Any, casting_context) -> Any:
        assert isinstance(value, tuple), f'Can not cast {value!r} to tuple type.'
        assert len(value) == len(self.components), f'Expected {value} to have length of {len(self.components)}'
        casted_values, was_casted = cast_and_return_whether_casted(self.components, value, casting_context)
        if was_casted:
            return tuple(casted_values)
        else:
            return value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, Tuple):
            return False
        return self.components == other.components

    def __hash__(self) -> int:
        return hash(self.components)

    def __repr__(self):
        return f'Tuple(components={self.components!r})'