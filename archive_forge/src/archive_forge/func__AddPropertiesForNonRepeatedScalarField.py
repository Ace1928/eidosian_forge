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
def _AddPropertiesForNonRepeatedScalarField(field, cls):
    """Adds a public property for a nonrepeated, scalar protocol message field.
  Clients can use this property to get and directly set the value of the field.
  Note that when the client sets the value of a field by using this property,
  all necessary "has" bits are set as a side-effect, and we also perform
  type-checking.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    proto_field_name = field.name
    property_name = _PropertyName(proto_field_name)
    type_checker = type_checkers.GetTypeChecker(field)
    default_value = field.default_value

    def getter(self):
        return self._fields.get(field, default_value)
    getter.__module__ = None
    getter.__doc__ = 'Getter for %s.' % proto_field_name

    def field_setter(self, new_value):
        try:
            new_value = type_checker.CheckValue(new_value)
        except TypeError as e:
            raise TypeError('Cannot set %s to %.1024r: %s' % (field.full_name, new_value, e))
        if not field.has_presence and (not new_value):
            self._fields.pop(field, None)
        else:
            self._fields[field] = new_value
        if not self._cached_byte_size_dirty:
            self._Modified()
    if field.containing_oneof:

        def setter(self, new_value):
            field_setter(self, new_value)
            self._UpdateOneofState(field)
    else:
        setter = field_setter
    setter.__module__ = None
    setter.__doc__ = 'Setter for %s.' % proto_field_name
    doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
    setattr(cls, property_name, _FieldProperty(field, getter, setter, doc=doc))