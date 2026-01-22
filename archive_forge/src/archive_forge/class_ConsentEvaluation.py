from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentEvaluation(_messages.Message):
    """The detailed evaluation of a particular Consent.

  Enums:
    EvaluationResultValueValuesEnum: The evaluation result.

  Fields:
    evaluationResult: The evaluation result.
  """

    class EvaluationResultValueValuesEnum(_messages.Enum):
        """The evaluation result.

    Values:
      EVALUATION_RESULT_UNSPECIFIED: No evaluation result specified. This
        option is invalid.
      NOT_APPLICABLE: The Consent is not applicable to the requested access
        determination. For example, the Consent does not apply to the user for
        which the access determination is requested, or it has a `state` of
        `REVOKED`, or it has expired.
      NO_MATCHING_POLICY: The Consent does not have a policy that matches the
        `resource_attributes` of the evaluated resource.
      NO_SATISFIED_POLICY: The Consent has at least one policy that matches
        the `resource_attributes` of the evaluated resource, but no
        `authorization_rule` was satisfied.
      HAS_SATISFIED_POLICY: The Consent has at least one policy that matches
        the `resource_attributes` of the evaluated resource, and at least one
        `authorization_rule` was satisfied.
    """
        EVALUATION_RESULT_UNSPECIFIED = 0
        NOT_APPLICABLE = 1
        NO_MATCHING_POLICY = 2
        NO_SATISFIED_POLICY = 3
        HAS_SATISFIED_POLICY = 4
    evaluationResult = _messages.EnumField('EvaluationResultValueValuesEnum', 1)