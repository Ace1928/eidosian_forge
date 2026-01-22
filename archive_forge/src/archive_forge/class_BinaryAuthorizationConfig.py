from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryAuthorizationConfig(_messages.Message):
    """BinaryAuthorizationConfig defines the fleet level configuration of
  binary authorization feature.

  Enums:
    EvaluationModeValueValuesEnum: Optional. Mode of operation for binauthz
      policy evaluation.

  Fields:
    evaluationMode: Optional. Mode of operation for binauthz policy
      evaluation.
    policyBindings: Optional. Binauthz policies that apply to this cluster.
  """

    class EvaluationModeValueValuesEnum(_messages.Enum):
        """Optional. Mode of operation for binauthz policy evaluation.

    Values:
      EVALUATION_MODE_UNSPECIFIED: Default value
      DISABLED: Disable BinaryAuthorization
      POLICY_BINDINGS: Use Binary Authorization with the policies specified in
        policy_bindings.
    """
        EVALUATION_MODE_UNSPECIFIED = 0
        DISABLED = 1
        POLICY_BINDINGS = 2
    evaluationMode = _messages.EnumField('EvaluationModeValueValuesEnum', 1)
    policyBindings = _messages.MessageField('PolicyBinding', 2, repeated=True)