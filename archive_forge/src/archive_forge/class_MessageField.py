import types
import weakref
import six
from apitools.base.protorpclite import util
class MessageField(Field):
    """Field definition for sub-message values.

    Message fields contain instance of other messages.  Instances stored
    on messages stored on message fields  are considered to be owned by
    the containing message instance and should not be shared between
    owning instances.

    Message fields must be defined to reference a single type of message.
    Normally message field are defined by passing the referenced message
    class in to the constructor.

    It is possible to define a message field for a type that does not
    yet exist by passing the name of the message in to the constructor
    instead of a message class. Resolution of the actual type of the
    message is deferred until it is needed, for example, during
    message verification. Names provided to the constructor must refer
    to a class within the same python module as the class that is
    using it. Names refer to messages relative to the containing
    messages scope. For example, the two fields of OuterMessage refer
    to the same message type:

      class Outer(Message):

        inner_relative = MessageField('Inner', 1)
        inner_absolute = MessageField('Outer.Inner', 2)

        class Inner(Message):
          ...

    When resolving an actual type, MessageField will traverse the
    entire scope of nested messages to match a message name. This
    makes it easy for siblings to reference siblings:

      class Outer(Message):

        class Inner(Message):

          sibling = MessageField('Sibling', 1)

        class Sibling(Message):
          ...

    """
    VARIANTS = frozenset([Variant.MESSAGE])
    DEFAULT_VARIANT = Variant.MESSAGE

    @util.positional(3)
    def __init__(self, message_type, number, required=False, repeated=False, variant=None):
        """Constructor.

        Args:
          message_type: Message type for field.  Must be subclass of Message.
          number: Number of field.  Must be unique per message class.
          required: Whether or not field is required.  Mutually exclusive to
            'repeated'.
          repeated: Whether or not field is repeated.  Mutually exclusive to
            'required'.
          variant: Wire-format variant hint.

        Raises:
          FieldDefinitionError when invalid message_type is provided.
        """
        valid_type = isinstance(message_type, six.string_types) or (message_type is not Message and isinstance(message_type, type) and issubclass(message_type, Message))
        if not valid_type:
            raise FieldDefinitionError('Invalid message class: %s' % message_type)
        if isinstance(message_type, six.string_types):
            self.__type_name = message_type
            self.__type = None
        else:
            self.__type = message_type
        super(MessageField, self).__init__(number, required=required, repeated=repeated, variant=variant)

    def __set__(self, message_instance, value):
        """Set value on message.

        Args:
          message_instance: Message instance to set value on.
          value: Value to set on message.
        """
        t = self.type
        if isinstance(t, type) and issubclass(t, Message):
            if self.repeated:
                if value and isinstance(value, (list, tuple)):
                    value = [t(**v) if isinstance(v, dict) else v for v in value]
            elif isinstance(value, dict):
                value = t(**value)
        super(MessageField, self).__set__(message_instance, value)

    @property
    def type(self):
        """Message type used for field."""
        if self.__type is None:
            message_type = find_definition(self.__type_name, self.message_definition())
            if not (message_type is not Message and isinstance(message_type, type) and issubclass(message_type, Message)):
                raise FieldDefinitionError('Invalid message class: %s' % message_type)
            self.__type = message_type
        return self.__type

    @property
    def message_type(self):
        """Underlying message type used for serialization.

        Will always be a sub-class of Message.  This is different from type
        which represents the python value that message_type is mapped to for
        use by the user.
        """
        return self.type

    def value_from_message(self, message):
        """Convert a message to a value instance.

        Used by deserializers to convert from underlying messages to
        value of expected user type.

        Args:
          message: A message instance of type self.message_type.

        Returns:
          Value of self.message_type.
        """
        if not isinstance(message, self.message_type):
            raise DecodeError('Expected type %s, got %s: %r' % (self.message_type.__name__, type(message).__name__, message))
        return message

    def value_to_message(self, value):
        """Convert a value instance to a message.

        Used by serializers to convert Python user types to underlying
        messages for transmission.

        Args:
          value: A value of type self.type.

        Returns:
          An instance of type self.message_type.
        """
        if not isinstance(value, self.type):
            raise EncodeError('Expected type %s, got %s: %r' % (self.type.__name__, type(value).__name__, value))
        return value