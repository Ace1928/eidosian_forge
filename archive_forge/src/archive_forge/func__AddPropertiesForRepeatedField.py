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
def _AddPropertiesForRepeatedField(field, cls):
    """Adds a public property for a "repeated" protocol message field.  Clients
  can use this property to get the value of the field, which will be either a
  RepeatedScalarFieldContainer or RepeatedCompositeFieldContainer (see
  below).

  Note that when clients add values to these containers, we perform
  type-checking in the case of repeated scalar fields, and we also set any
  necessary "has" bits as a side-effect.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    proto_field_name = field.name
    property_name = _PropertyName(proto_field_name)

    def getter(self):
        field_value = self._fields.get(field)
        if field_value is None:
            field_value = field._default_constructor(self)
            field_value = self._fields.setdefault(field, field_value)
        return field_value
    getter.__module__ = None
    getter.__doc__ = 'Getter for %s.' % proto_field_name

    def setter(self, new_value):
        raise AttributeError('Assignment not allowed to repeated field "%s" in protocol message object.' % proto_field_name)
    doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
    setattr(cls, property_name, _FieldProperty(field, getter, setter, doc=doc))