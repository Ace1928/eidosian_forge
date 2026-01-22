from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def ByteSize(self):
    if not self._cached_byte_size_dirty:
        return self._cached_byte_size
    size = 0
    descriptor = self.DESCRIPTOR
    if descriptor._is_map_entry:
        key_field = descriptor.fields_by_name['key']
        _MaybeAddEncoder(cls, key_field)
        size = key_field._sizer(self.key)
        value_field = descriptor.fields_by_name['value']
        _MaybeAddEncoder(cls, value_field)
        size += value_field._sizer(self.value)
    else:
        for field_descriptor, field_value in self.ListFields():
            _MaybeAddEncoder(cls, field_descriptor)
            size += field_descriptor._sizer(field_value)
        for tag_bytes, value_bytes in self._unknown_fields:
            size += len(tag_bytes) + len(value_bytes)
    self._cached_byte_size = size
    self._cached_byte_size_dirty = False
    self._listener_for_children.dirty = False
    return size