from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EvaluationRule(_messages.Message):
    """An evaluation rule specifies either that all container images used in a
  deployment request must be attested to by one or more Attestor, that the
  deployment will be always allowed, or that it is always denied.

  Enums:
    EnforcementModeValueValuesEnum: Required. Define the possible actions when
      a deployment is denied by an evaluation rule.
    EvaluationModeValueValuesEnum: Required. How this rule will be evaluated.

  Fields:
    enforcementMode: Required. Define the possible actions when a deployment
      is denied by an evaluation rule.
    evaluationMode: Required. How this rule will be evaluated.
    requiredAttestors: Optional. If the `evaluation_mode` is
      `REQUIRE_ATTESTATION`, this is the list of the attestors required for
      the deployment.
  """

    class EnforcementModeValueValuesEnum(_messages.Enum):
        """Required. Define the possible actions when a deployment is denied by
    an evaluation rule.

    Values:
      ENFORCEMENT_MODE_UNSPECIFIED: Do not use.
      ENFORCED_BLOCK_AND_AUDIT_LOG: Enforce the admission rule by blocking the
        deployment.
      DRYRUN_AUDIT_LOG_ONLY: Dryrun mode: Audit logging only. This will allow
        the deployment as if the admission request had specified break-glass.
    """
        ENFORCEMENT_MODE_UNSPECIFIED = 0
        ENFORCED_BLOCK_AND_AUDIT_LOG = 1
        DRYRUN_AUDIT_LOG_ONLY = 2

    class EvaluationModeValueValuesEnum(_messages.Enum):
        """Required. How this rule will be evaluated.

    Values:
      EVALUATION_MODE_UNSPECIFIED: Do not use.
      ALWAYS_ALLOW: The deployment is always allowed.
      REQUIRE_ATTESTATION: The deployment requires attestations from certain
        attestors.
      ALWAYS_DENY: The deployment is always denied.
    """
        EVALUATION_MODE_UNSPECIFIED = 0
        ALWAYS_ALLOW = 1
        REQUIRE_ATTESTATION = 2
        ALWAYS_DENY = 3
    enforcementMode = _messages.EnumField('EnforcementModeValueValuesEnum', 1)
    evaluationMode = _messages.EnumField('EvaluationModeValueValuesEnum', 2)
    requiredAttestors = _messages.MessageField('InlineAttestor', 3, repeated=True)