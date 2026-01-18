import abc
from .struct import Struct
from .types import Int16, Int32, String, Schema, Array, TaggedFields
def to_object(self):
    return _to_object(self.SCHEMA, self)