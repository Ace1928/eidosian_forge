from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientConnectorService(_messages.Message):
    """Message describing ClientConnectorService object.

  Enums:
    StateValueValuesEnum: Output only. The operational state of the
      ClientConnectorService.

  Fields:
    createTime: Output only. [Output only] Create time stamp.
    displayName: Optional. User-provided name. The display name should follow
      certain format. * Must be 6 to 30 characters in length. * Can only
      contain lowercase letters, numbers, and hyphens. * Must start with a
      letter.
    egress: Required. The details of the egress settings.
    ingress: Required. The details of the ingress settings.
    name: Required. Name of resource. The name is ignored during creation.
    state: Output only. The operational state of the ClientConnectorService.
    updateTime: Output only. [Output only] Update time stamp.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The operational state of the ClientConnectorService.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      CREATING: ClientConnectorService is being created.
      UPDATING: ClientConnectorService is being updated.
      DELETING: ClientConnectorService is being deleted.
      RUNNING: ClientConnectorService is running.
      DOWN: ClientConnectorService is down and may be restored in the future.
        This happens when CCFE sends ProjectState = OFF.
      ERROR: ClientConnectorService encountered an error and is in an
        indeterministic state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        UPDATING = 2
        DELETING = 3
        RUNNING = 4
        DOWN = 5
        ERROR = 6
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    egress = _messages.MessageField('Egress', 3)
    ingress = _messages.MessageField('Ingress', 4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    updateTime = _messages.StringField(7)