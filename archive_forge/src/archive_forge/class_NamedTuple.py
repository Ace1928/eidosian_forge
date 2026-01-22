import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class NamedTuple(trace.TraceType, serialization.Serializable):
    """Represents a NamedTuple of TraceType objects."""

    def __init__(self, type_name: str, attribute_names: PythonTuple[str], attributes: PythonTuple[trace.TraceType], placeholder_type: Optional[Type[Any]]=None):
        self.type_name = type_name
        self.attribute_names = attribute_names
        self.attributes = Tuple(*attributes)
        self._placeholder_type = placeholder_type

    @classmethod
    def from_type_and_attributes(cls, named_tuple_type: Any, attributes: PythonTuple[trace.TraceType]) -> 'NamedTuple':
        return NamedTuple(named_tuple_type.__name__, named_tuple_type._fields, attributes, named_tuple_type)

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        if not isinstance(other, NamedTuple):
            return False
        return self.type_name == other.type_name and self.attribute_names == other.attribute_names and self.attributes.is_subtype_of(other.attributes)

    def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional['NamedTuple']:
        """See base class."""
        if not all((isinstance(other, NamedTuple) and self.type_name == other.type_name and (self.attribute_names == other.attribute_names) for other in others)):
            return None
        supertyped_attributes = self.attributes.most_specific_common_supertype([other.attributes for other in others])
        if supertyped_attributes is None:
            return None
        return NamedTuple(self.type_name, self.attribute_names, supertyped_attributes.components, self._placeholder_type)

    @classmethod
    def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedNamedTuple]:
        return default_types_pb2.SerializedNamedTuple

    @classmethod
    def experimental_from_proto(cls, proto: default_types_pb2.SerializedNamedTuple) -> 'NamedTuple':
        return NamedTuple(proto.type_name, tuple(proto.attribute_names), Tuple.experimental_from_proto(proto.attributes).components)

    def experimental_as_proto(self) -> default_types_pb2.SerializedNamedTuple:
        return default_types_pb2.SerializedNamedTuple(type_name=self.type_name, attribute_names=list(self.attribute_names), attributes=self.attributes.experimental_as_proto())

    def placeholder_value(self, placeholder_context) -> Any:
        if self._placeholder_type is None:
            raise ValueError('Can not generate placeholder value for NamedTuple with unspecified placeholder_type. Note: placeholder_type is lost during serialization.')
        attribute_placeholders = [attribute.placeholder_value(placeholder_context) for attribute in self.attributes.components]
        return self._placeholder_type(*attribute_placeholders)

    def _to_tensors(self, value: Any):
        assert util.is_namedtuple(value)
        flattened_values = []
        for attribute_name, attribute_type in zip(self.attribute_names, self.attributes.components):
            attribute_value = getattr(value, attribute_name)
            flattened_values.extend(attribute_type._to_tensors(attribute_value))
        return flattened_values

    def _from_tensors(self, tensors) -> Any:
        if self._placeholder_type is None:
            raise ValueError('Packing serialized NamedTuples is not supported.')
        return self._placeholder_type(*[c._from_tensors(tensors) for c in self.attributes.components])

    def _flatten(self) -> PythonList[trace.TraceType]:
        flattened_types = []
        for component in self.attributes.components:
            flattened_types.extend(component._flatten())
        return flattened_types

    def _cast(self, value: Any, casting_context) -> Any:
        assert util.is_namedtuple(value), f'Cannot cast {value!r} to type {self._placeholder_type!r}.'
        value_dict = value._asdict()
        assert set(value_dict.keys()) == set(self.attribute_names), f'{value!r} has different attributes with the TraceType {self!r}'
        casted_values, was_casted = cast_and_return_whether_casted(self.attributes.components, [getattr(value, name) for name in self.attribute_names], casting_context)
        if was_casted:
            return self._placeholder_type(*casted_values)
        else:
            return value

    def __hash__(self) -> int:
        return hash((self.type_name, self.attribute_names, self.attributes))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, NamedTuple):
            return False
        return self.type_name == other.type_name and self.attribute_names == other.attribute_names and (self.attributes == other.attributes)

    def __repr__(self):
        return f'NamedTuple(type_name={self.type_name}, attribute_names={self.attribute_names}, attributes={self.attributes.components})'