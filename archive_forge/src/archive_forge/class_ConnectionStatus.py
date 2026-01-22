from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionStatus(_messages.Message):
    """ConnectionStatus indicates the state of the connection.

  Enums:
    StateValueValuesEnum: State.

  Fields:
    description: Description.
    state: State.
    status: Status provides detailed information for the state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State.

    Values:
      STATE_UNSPECIFIED: Connection does not have a state yet.
      CREATING: Connection is being created.
      ACTIVE: Connection is running and ready for requests.
      INACTIVE: Connection is stopped.
      DELETING: Connection is being deleted.
      UPDATING: Connection is being updated.
      ERROR: Connection is not running due to an error.
      AUTHORIZATION_REQUIRED: Connection is not running because the
        authorization configuration is not complete.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        INACTIVE = 3
        DELETING = 4
        UPDATING = 5
        ERROR = 6
        AUTHORIZATION_REQUIRED = 7
    description = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    status = _messages.StringField(3)