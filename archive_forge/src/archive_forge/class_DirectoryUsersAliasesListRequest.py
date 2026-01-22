from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersAliasesListRequest(_messages.Message):
    """A DirectoryUsersAliasesListRequest object.

  Enums:
    EventValueValuesEnum: Event on which subscription is intended (if
      subscribing)

  Fields:
    event: Event on which subscription is intended (if subscribing)
    userKey: Email or immutable ID of the user
  """

    class EventValueValuesEnum(_messages.Enum):
        """Event on which subscription is intended (if subscribing)

    Values:
      add: Alias Created Event
      delete: Alias Deleted Event
    """
        add = 0
        delete = 1
    event = _messages.EnumField('EventValueValuesEnum', 1)
    userKey = _messages.StringField(2, required=True)