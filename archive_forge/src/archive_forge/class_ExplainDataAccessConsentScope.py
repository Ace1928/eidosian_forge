from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainDataAccessConsentScope(_messages.Message):
    """A single consent scope that provides info on who has access to the
  requested resource scope for a particular purpose and environment, enforced
  by which consent.

  Enums:
    DecisionValueValuesEnum: Whether the current consent scope is permitted or
      denied access on the requested resource.

  Fields:
    accessorScope: The accessor scope that describes who can access, for what
      purpose, and in which environment.
    decision: Whether the current consent scope is permitted or denied access
      on the requested resource.
    enforcingConsents: Metadata of the consent resources that enforce the
      consent scope's access.
    exceptions: Other consent scopes that created exceptions within this
      scope.
  """

    class DecisionValueValuesEnum(_messages.Enum):
        """Whether the current consent scope is permitted or denied access on the
    requested resource.

    Values:
      CONSENT_DECISION_TYPE_UNSPECIFIED: Unspecified consent decision type.
      CONSENT_DECISION_TYPE_PERMIT: Consent permitted access.
      CONSENT_DECISION_TYPE_DENY: Consent denied access.
    """
        CONSENT_DECISION_TYPE_UNSPECIFIED = 0
        CONSENT_DECISION_TYPE_PERMIT = 1
        CONSENT_DECISION_TYPE_DENY = 2
    accessorScope = _messages.MessageField('ConsentAccessorScope', 1)
    decision = _messages.EnumField('DecisionValueValuesEnum', 2)
    enforcingConsents = _messages.MessageField('ExplainDataAccessConsentInfo', 3, repeated=True)
    exceptions = _messages.MessageField('ExplainDataAccessConsentScope', 4, repeated=True)