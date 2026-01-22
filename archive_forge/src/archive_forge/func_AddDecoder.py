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
def AddDecoder(is_packed):
    decode_type = field_descriptor.type
    if decode_type == _FieldDescriptor.TYPE_ENUM and (not field_descriptor.enum_type.is_closed):
        decode_type = _FieldDescriptor.TYPE_INT32
    oneof_descriptor = None
    if field_descriptor.containing_oneof is not None:
        oneof_descriptor = field_descriptor
    if is_map_entry:
        is_message_map = _IsMessageMapField(field_descriptor)
        field_decoder = decoder.MapDecoder(field_descriptor, _GetInitializeDefaultForMap(field_descriptor), is_message_map)
    elif decode_type == _FieldDescriptor.TYPE_STRING:
        field_decoder = decoder.StringDecoder(field_descriptor.number, is_repeated, is_packed, field_descriptor, field_descriptor._default_constructor, not field_descriptor.has_presence)
    elif field_descriptor.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        field_decoder = type_checkers.TYPE_TO_DECODER[decode_type](field_descriptor.number, is_repeated, is_packed, field_descriptor, field_descriptor._default_constructor)
    else:
        field_decoder = type_checkers.TYPE_TO_DECODER[decode_type](field_descriptor.number, is_repeated, is_packed, field_descriptor, field_descriptor._default_constructor, not field_descriptor.has_presence)
    helper_decoders[is_packed] = field_decoder