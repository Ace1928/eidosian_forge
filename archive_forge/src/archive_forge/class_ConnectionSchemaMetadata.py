from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionSchemaMetadata(_messages.Message):
    """ConnectionSchemaMetadata is the singleton resource of each connection.
  It includes the entity and action names of runtime resources exposed by a
  connection backend.

  Enums:
    StateValueValuesEnum: Output only. The current state of runtime schema.

  Fields:
    actions: Output only. List of actions.
    entities: Output only. List of entity names.
    name: Output only. Resource name. Format: projects/{project}/locations/{lo
      cation}/connections/{connection}/connectionSchemaMetadata
    refreshTime: Output only. Timestamp when the connection runtime schema
      refresh was triggered.
    state: Output only. The current state of runtime schema.
    updateTime: Output only. Timestamp when the connection runtime schema was
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of runtime schema.

    Values:
      STATE_UNSPECIFIED: Default state.
      REFRESHING: Schema refresh is in progress.
      UPDATED: Schema has been updated.
    """
        STATE_UNSPECIFIED = 0
        REFRESHING = 1
        UPDATED = 2
    actions = _messages.StringField(1, repeated=True)
    entities = _messages.StringField(2, repeated=True)
    name = _messages.StringField(3)
    refreshTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    updateTime = _messages.StringField(6)