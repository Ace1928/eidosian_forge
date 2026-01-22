from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Session(_messages.Message):
    """Represents an active analyze session running for a user.

  Enums:
    StateValueValuesEnum: Output only. State of Session

  Fields:
    createTime: Output only. Session start time.
    name: Output only. The relative resource name of the content, of the form:
      projects/{project_id}/locations/{location_id}/lakes/{lake_id}/environmen
      t/{environment_id}/sessions/{session_id}
    state: Output only. State of Session
    userId: Output only. Email of user running the session.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of Session

    Values:
      STATE_UNSPECIFIED: State is not specified.
      ACTIVE: Resource is active, i.e., ready to use.
      CREATING: Resource is under creation.
      DELETING: Resource is under deletion.
      ACTION_REQUIRED: Resource is active but has unresolved actions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        ACTION_REQUIRED = 4
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    userId = _messages.StringField(4)