from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignmentReportOSPolicyComplianceOSPolicyResourceComplianceOSPolicyResourceConfigStep(_messages.Message):
    """Step performed by the OS Config agent for configuring an `OSPolicy`
  resource to its desired state.

  Enums:
    TypeValueValuesEnum: Configuration step type.

  Fields:
    errorMessage: An error message recorded during the execution of this step.
      Only populated if errors were encountered during this step execution.
    type: Configuration step type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Configuration step type.

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      VALIDATION: Checks for resource conflicts such as schema errors.
      DESIRED_STATE_CHECK: Checks the current status of the desired state for
        a resource.
      DESIRED_STATE_ENFORCEMENT: Enforces the desired state for a resource
        that is not in desired state.
      DESIRED_STATE_CHECK_POST_ENFORCEMENT: Re-checks the status of the
        desired state. This check is done for a resource after the enforcement
        of all OS policies. This step is used to determine the final desired
        state status for the resource. It accounts for any resources that
        might have drifted from their desired state due to side effects from
        executing other resources.
    """
        TYPE_UNSPECIFIED = 0
        VALIDATION = 1
        DESIRED_STATE_CHECK = 2
        DESIRED_STATE_ENFORCEMENT = 3
        DESIRED_STATE_CHECK_POST_ENFORCEMENT = 4
    errorMessage = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)