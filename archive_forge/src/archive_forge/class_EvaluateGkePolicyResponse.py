from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluateGkePolicyResponse(_messages.Message):
    """Response message for PlatformPolicyEvaluationService.EvaluateGkePolicy.

  Enums:
    VerdictValueValuesEnum: The result of evaluating all Pods in the request.

  Fields:
    attestations: If AttestationMode is set to `GENERATE_DEPLOY` and the top-
      level verdict is conformant, an attestation will be returned for each
      image in the request. Attestations are in the form of websafe base64
      encoded JSON DSSEs (https://github.com/secure-systems-
      lab/dsse/blob/master/envelope.md).
    results: Evaluation result for each Pod contained in the request.
    verdict: The result of evaluating all Pods in the request.
  """

    class VerdictValueValuesEnum(_messages.Enum):
        """The result of evaluating all Pods in the request.

    Values:
      VERDICT_UNSPECIFIED: Not specified. This should never be used.
      CONFORMANT: All Pods in the request conform to the policy.
      NON_CONFORMANT: At least one Pod does not conform to the policy.
      ERROR: Encountered at least one error evaluating a Pod and all other
        Pods conform to the policy. Non-conformance has precedence over
        errors.
    """
        VERDICT_UNSPECIFIED = 0
        CONFORMANT = 1
        NON_CONFORMANT = 2
        ERROR = 3
    attestations = _messages.StringField(1, repeated=True)
    results = _messages.MessageField('PodResult', 2, repeated=True)
    verdict = _messages.EnumField('VerdictValueValuesEnum', 3)