import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
class Attrs(trace.TraceType):
    """Represents a class annotated by attr.s."""

    def __init__(self, type_name: str, attribute_names: PythonTuple[str], attributes: PythonTuple[trace.TraceType], placeholder_type: Optional[Type[Any]]=None):
        self.named_attributes = NamedTuple(type_name, attribute_names, attributes)
        self._placeholder_type = placeholder_type

    @classmethod
    def from_type_and_attributes(cls, attrs_type: Any, attributes: PythonTuple[trace.TraceType]) -> 'Attrs':
        return Attrs(attrs_type.__name__, tuple((attr.name for attr in attrs_type.__attrs_attrs__)), attributes, attrs_type)

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        if not isinstance(other, Attrs):
            return False
        return self.named_attributes.is_subtype_of(other.named_attributes)

    def most_specific_common_supertype(self, others: Sequence[trace.TraceType]) -> Optional['Attrs']:
        """See base class."""
        if not all((isinstance(other, Attrs) for other in others)):
            return None
        supertyped_attributes = self.named_attributes.most_specific_common_supertype([other.named_attributes for other in others])
        if supertyped_attributes is None:
            return None
        return Attrs(self.named_attributes.type_name, self.named_attributes.attribute_names, supertyped_attributes.attributes.components, self._placeholder_type)

    @classmethod
    def experimental_type_proto(cls) -> Type[default_types_pb2.SerializedAttrs]:
        return default_types_pb2.SerializedAttrs

    @classmethod
    def experimental_from_proto(cls, proto: default_types_pb2.SerializedAttrs) -> 'Attrs':
        return Attrs(proto.named_attributes.type_name, tuple(proto.named_attributes.attribute_names), Tuple.experimental_from_proto(proto.named_attributes.attributes).components)

    def experimental_as_proto(self) -> default_types_pb2.SerializedAttrs:
        return default_types_pb2.SerializedAttrs(named_attributes=self.named_attributes.experimental_as_proto())

    def placeholder_value(self, placeholder_context) -> Any:
        if self._placeholder_type is None:
            raise ValueError('Can not generate placeholder value for Attrs with unspecified placeholder_type. Note: placeholder_type is lost during serialization.')
        attribute_placeholders = [attribute.placeholder_value(placeholder_context) for attribute in self.named_attributes.attributes.components]
        return self._placeholder_type(*attribute_placeholders)

    def _to_tensors(self, value: Any):
        assert util.is_attrs(value)
        flattened_values = []
        for attribute_name, attribute_type in zip(self.named_attributes.attribute_names, self.named_attributes.attributes.components):
            attribute_value = getattr(value, attribute_name)
            flattened_values.extend(attribute_type._to_tensors(attribute_value))
        return flattened_values

    def _from_tensors(self, tensors):
        if self._placeholder_type is None:
            raise ValueError('Packing serialized NamedTuples is not supported.')
        return self._placeholder_type(*[c._from_tensors(tensors) for c in self.named_attributes.attributes.components])

    def _flatten(self) -> PythonList[trace.TraceType]:
        flattened_types = []
        for component in self.named_attributes.attributes.components:
            flattened_types.extend(component._flatten())
        return flattened_types

    def _cast(self, value: Any, casting_context) -> Any:
        assert util.is_attrs(value)
        attr_names = self.named_attributes.attribute_names
        casted_values, was_casted = cast_and_return_whether_casted(self.named_attributes.attributes.components, [getattr(value, name) for name in attr_names], casting_context)
        if was_casted:
            return self._placeholder_type(*casted_values)
        else:
            return value

    def __hash__(self) -> int:
        return hash(self.named_attributes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, trace.TraceType):
            return NotImplemented
        if not isinstance(other, Attrs):
            return False
        return self.named_attributes == other.named_attributes

    def __repr__(self):
        return f'Attrs(type_name={self.named_attributes.type_name}, attribute_names={self.named_attributes.attribute_names}, attributes={self.named_attributes.attributes.components})'