from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AgentPool(_messages.Message):
    """Represents an On-Premises Agent pool.

  Enums:
    StateValueValuesEnum: Output only. Specifies the state of the AgentPool.

  Fields:
    bandwidthLimit: Specifies the bandwidth limit details. If this field is
      unspecified, the default value is set as 'No Limit'.
    displayName: Specifies the client-specified AgentPool description.
    name: Required. Specifies a unique string that identifies the agent pool.
      Format: `projects/{project_id}/agentPools/{agent_pool_id}`
    state: Output only. Specifies the state of the AgentPool.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Specifies the state of the AgentPool.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      CREATING: This is an initialization state. During this stage, the
        resources such as Pub/Sub topics are allocated for the AgentPool.
      CREATED: Determines that the AgentPool is created for use. At this
        state, Agents can join the AgentPool and participate in the transfer
        jobs in that pool.
      DELETING: Determines that the AgentPool deletion has been initiated, and
        all the resources are scheduled to be cleaned up and freed.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        CREATED = 2
        DELETING = 3
    bandwidthLimit = _messages.MessageField('BandwidthLimit', 1)
    displayName = _messages.StringField(2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)