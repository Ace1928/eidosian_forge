from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceMessage(_messages.Message):
    """A InstanceMessage object.

  Enums:
    CodeValueValuesEnum: A code that correspond to one type of user-facing
      message.

  Fields:
    code: A code that correspond to one type of user-facing message.
    message: Message on memcached instance which will be exposed to users.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """A code that correspond to one type of user-facing message.

    Values:
      CODE_UNSPECIFIED: Message Code not set.
      ZONE_DISTRIBUTION_UNBALANCED: Memcached nodes are distributed unevenly.
    """
        CODE_UNSPECIFIED = 0
        ZONE_DISTRIBUTION_UNBALANCED = 1
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    message = _messages.StringField(2)