from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionState(_messages.Message):
    """ConnectionState holds the current connection state from the cluster to
  Google.

  Enums:
    StateValueValuesEnum: Output only. The current connection state.

  Fields:
    state: Output only. The current connection state.
    updateTime: Output only. The time when the connection state was last
      changed.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current connection state.

    Values:
      STATE_UNSPECIFIED: Unknown connection state.
      DISCONNECTED: This cluster is currently disconnected from Google.
      CONNECTED: This cluster is currently connected to Google.
      CONNECTED_AND_SYNCING: This cluster is currently connected to Google,
        but may have recently reconnected after a disconnection. It is still
        syncing back.
    """
        STATE_UNSPECIFIED = 0
        DISCONNECTED = 1
        CONNECTED = 2
        CONNECTED_AND_SYNCING = 3
    state = _messages.EnumField('StateValueValuesEnum', 1)
    updateTime = _messages.StringField(2)