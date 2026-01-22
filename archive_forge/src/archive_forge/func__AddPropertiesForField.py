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
def _AddPropertiesForField(field, cls):
    """Adds a public property for a protocol message field.
  Clients can use this property to get and (in the case
  of non-repeated scalar fields) directly set the value
  of a protocol message field.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    assert _FieldDescriptor.MAX_CPPTYPE == 10
    constant_name = field.name.upper() + '_FIELD_NUMBER'
    setattr(cls, constant_name, field.number)
    if field.label == _FieldDescriptor.LABEL_REPEATED:
        _AddPropertiesForRepeatedField(field, cls)
    elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        _AddPropertiesForNonRepeatedCompositeField(field, cls)
    else:
        _AddPropertiesForNonRepeatedScalarField(field, cls)