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
def _GetInitializeDefaultForMap(field):
    if field.label != _FieldDescriptor.LABEL_REPEATED:
        raise ValueError('map_entry set on non-repeated field %s' % field.name)
    fields_by_name = field.message_type.fields_by_name
    key_checker = type_checkers.GetTypeChecker(fields_by_name['key'])
    value_field = fields_by_name['value']
    if _IsMessageMapField(field):

        def MakeMessageMapDefault(message):
            return containers.MessageMap(message._listener_for_children, value_field.message_type, key_checker, field.message_type)
        return MakeMessageMapDefault
    else:
        value_checker = type_checkers.GetTypeChecker(value_field)

        def MakePrimitiveMapDefault(message):
            return containers.ScalarMap(message._listener_for_children, key_checker, value_checker, field.message_type)
        return MakePrimitiveMapDefault