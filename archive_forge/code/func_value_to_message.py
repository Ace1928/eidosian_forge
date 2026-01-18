import types
import weakref
import six
from apitools.base.protorpclite import util
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