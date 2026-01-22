import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class MessageDescriptor(messages.Message):
    """Message definition descriptor.

    Fields:
      name: Name of Message without any qualification.
      fields: Fields defined for message.
      message_types: Nested Message classes defined on message.
      enum_types: Nested Enum classes defined on message.
    """
    name = messages.StringField(1)
    fields = messages.MessageField(FieldDescriptor, 2, repeated=True)
    message_types = messages.MessageField('apitools.base.protorpclite.descriptor.MessageDescriptor', 3, repeated=True)
    enum_types = messages.MessageField(EnumDescriptor, 4, repeated=True)