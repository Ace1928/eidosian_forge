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
def InternalParse(self, buffer, pos, end):
    """Create a message from serialized bytes.

    Args:
      self: Message, instance of the proto message object.
      buffer: memoryview of the serialized data.
      pos: int, position to start in the serialized data.
      end: int, end position of the serialized data.

    Returns:
      Message object.
    """
    assert isinstance(buffer, memoryview)
    self._Modified()
    field_dict = self._fields
    unknown_field_set = self._unknown_field_set
    while pos != end:
        tag_bytes, new_pos = local_ReadTag(buffer, pos)
        field_decoder, field_des = message_set_decoders_by_tag.get(tag_bytes, (None, None))
        if field_decoder:
            pos = field_decoder(buffer, new_pos, end, self, field_dict)
            continue
        field_des, is_packed = fields_by_tag.get(tag_bytes, (None, None))
        if field_des is None:
            if not self._unknown_fields:
                self._unknown_fields = []
            if unknown_field_set is None:
                self._unknown_field_set = containers.UnknownFieldSet()
                unknown_field_set = self._unknown_field_set
            tag, _ = decoder._DecodeVarint(tag_bytes, 0)
            field_number, wire_type = wire_format.UnpackTag(tag)
            if field_number == 0:
                raise message_mod.DecodeError('Field number 0 is illegal.')
            old_pos = new_pos
            data, new_pos = decoder._DecodeUnknownField(buffer, new_pos, wire_type)
            if new_pos == -1:
                return pos
            unknown_field_set._add(field_number, wire_type, data)
            new_pos = local_SkipField(buffer, old_pos, end, tag_bytes)
            if new_pos == -1:
                return pos
            self._unknown_fields.append((tag_bytes, buffer[old_pos:new_pos].tobytes()))
            pos = new_pos
        else:
            _MaybeAddDecoder(cls, field_des)
            field_decoder = field_des._decoders[is_packed]
            pos = field_decoder(buffer, new_pos, end, self, field_dict)
            if field_des.containing_oneof:
                self._UpdateOneofState(field_des)
    return pos