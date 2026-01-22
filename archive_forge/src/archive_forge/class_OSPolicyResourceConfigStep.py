from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceConfigStep(_messages.Message):
    """Step performed by the OS Config agent for configuring an
  `OSPolicyResource` to its desired state.

  Enums:
    OutcomeValueValuesEnum: Outcome of the configuration step.
    TypeValueValuesEnum: Configuration step type.

  Fields:
    errorMessage: An error message recorded during the execution of this step.
      Only populated when outcome is FAILED.
    outcome: Outcome of the configuration step.
    type: Configuration step type.
  """

    class OutcomeValueValuesEnum(_messages.Enum):
        """Outcome of the configuration step.

    Values:
      OUTCOME_UNSPECIFIED: Default value. This value is unused.
      SUCCEEDED: The step succeeded.
      FAILED: The step failed.
    """
        OUTCOME_UNSPECIFIED = 0
        SUCCEEDED = 1
        FAILED = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Configuration step type.

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      VALIDATION: Validation to detect resource conflicts, schema errors, etc.
      DESIRED_STATE_CHECK: Check the current desired state status of the
        resource.
      DESIRED_STATE_ENFORCEMENT: Enforce the desired state for a resource that
        is not in desired state.
      DESIRED_STATE_CHECK_POST_ENFORCEMENT: Re-check desired state status for
        a resource after enforcement of all resources in the current
        configuration run. This step is used to determine the final desired
        state status for the resource. It accounts for any resources that
        might have drifted from their desired state due to side effects from
        configuring other resources during the current configuration run.
    """
        TYPE_UNSPECIFIED = 0
        VALIDATION = 1
        DESIRED_STATE_CHECK = 2
        DESIRED_STATE_ENFORCEMENT = 3
        DESIRED_STATE_CHECK_POST_ENFORCEMENT = 4
    errorMessage = _messages.StringField(1)
    outcome = _messages.EnumField('OutcomeValueValuesEnum', 2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)