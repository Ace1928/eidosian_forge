from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OSPolicyAssignmentOperationMetadata(_messages.Message):
    """OS policy assignment operation metadata provided by OS policy assignment
  API methods that return long running operations.

  Enums:
    ApiMethodValueValuesEnum: The OS policy assignment API method.
    RolloutStateValueValuesEnum: State of the rollout

  Fields:
    apiMethod: The OS policy assignment API method.
    osPolicyAssignment: Reference to the `OSPolicyAssignment` API resource.
      Format: `projects/{project_number}/locations/{location}/osPolicyAssignme
      nts/{os_policy_assignment_id@revision_id}`
    rolloutStartTime: Rollout start time
    rolloutState: State of the rollout
    rolloutUpdateTime: Rollout update time
  """

    class ApiMethodValueValuesEnum(_messages.Enum):
        """The OS policy assignment API method.

    Values:
      API_METHOD_UNSPECIFIED: Invalid value
      CREATE: Create OS policy assignment API method
      UPDATE: Update OS policy assignment API method
      DELETE: Delete OS policy assignment API method
    """
        API_METHOD_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
        DELETE = 3

    class RolloutStateValueValuesEnum(_messages.Enum):
        """State of the rollout

    Values:
      ROLLOUT_STATE_UNSPECIFIED: Invalid value
      IN_PROGRESS: The rollout is in progress.
      CANCELLING: The rollout is being cancelled.
      CANCELLED: The rollout is cancelled.
      SUCCEEDED: The rollout has completed successfully.
    """
        ROLLOUT_STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        CANCELLING = 2
        CANCELLED = 3
        SUCCEEDED = 4
    apiMethod = _messages.EnumField('ApiMethodValueValuesEnum', 1)
    osPolicyAssignment = _messages.StringField(2)
    rolloutStartTime = _messages.StringField(3)
    rolloutState = _messages.EnumField('RolloutStateValueValuesEnum', 4)
    rolloutUpdateTime = _messages.StringField(5)