from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSecuritygatewaysV1alphaSecurityGateway(_messages.Message):
    """Information about a BeyoncCorp SecurityGateway resource.

  Enums:
    StateValueValuesEnum: Output only. The operational state of the
      SecurityGateway.

  Fields:
    createTime: Output only. Timestamp when the resource was created.
    displayName: Optional. An arbitrary user-provided name for the
      SecurityGateway. Cannot exceed 64 characters.
    egressIpAddresses: Output only. IP addresses that will be used for
      establishing connection to the egress endpoints.
    name: Identifier. Name of the resource.
    state: Output only. The operational state of the SecurityGateway.
    updateTime: Output only. Timestamp when the resource was last modified.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The operational state of the SecurityGateway.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      CREATING: SecurityGateway is being created.
      UPDATING: SecurityGateway is being updated.
      DELETING: SecurityGateway is being deleted.
      RUNNING: SecurityGateway is running.
      DOWN: SecurityGateway is down and may be restored in the future. This
        happens when CCFE sends ProjectState = OFF.
      ERROR: SecurityGateway encountered an error and is in an indeterministic
        state.
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
    egressIpAddresses = _messages.StringField(3, repeated=True)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    updateTime = _messages.StringField(6)