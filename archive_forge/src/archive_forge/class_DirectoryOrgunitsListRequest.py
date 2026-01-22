from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryOrgunitsListRequest(_messages.Message):
    """A DirectoryOrgunitsListRequest object.

  Enums:
    TypeValueValuesEnum: Whether to return all sub-organizations or just
      immediate children

  Fields:
    customerId: Immutable ID of the G Suite account
    orgUnitPath: the URL-encoded organizational unit's path or its ID
    type: Whether to return all sub-organizations or just immediate children
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Whether to return all sub-organizations or just immediate children

    Values:
      all: All sub-organizational units.
      children: Immediate children only (default).
    """
        all = 0
        children = 1
    customerId = _messages.StringField(1, required=True)
    orgUnitPath = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)