import types
import weakref
import six
from apitools.base.protorpclite import util
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