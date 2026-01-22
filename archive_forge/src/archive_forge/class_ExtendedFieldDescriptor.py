import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class ExtendedFieldDescriptor(messages.Message):
    """Field descriptor with additional fields.

    Fields:
      field_descriptor: The underlying field descriptor.
      name: The name of this field.
      description: Description of this field.
    """
    field_descriptor = messages.MessageField(protorpc_descriptor.FieldDescriptor, 100)
    name = messages.StringField(101)
    description = messages.StringField(102)