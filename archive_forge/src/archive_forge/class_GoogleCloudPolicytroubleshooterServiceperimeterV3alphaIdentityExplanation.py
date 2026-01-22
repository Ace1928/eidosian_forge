from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaIdentityExplanation(_messages.Message):
    """Explanation of an identity. NextTAG: 3

  Enums:
    IdentityEvalStateValueValuesEnum: Output only. Details about the
      evaluation state of the identity set in policy.
    IdentityStateValueValuesEnum: Output only. The current state of an
      identity set in policy.

  Fields:
    identityEvalState: Output only. Details about the evaluation state of the
      identity set in policy.
    identityState: Output only. The current state of an identity set in
      policy.
  """

    class IdentityEvalStateValueValuesEnum(_messages.Enum):
        """Output only. Details about the evaluation state of the identity set in
    policy.

    Values:
      IDENTITY_EVAL_STATE_UNSPECIFIED: Not used
      MATCH: The request matches the identity
      NOT_MATCH: The request doesn't match the identity
      NOT_SUPPORTED: The identity is not supported
      INFO_DENIED: The sender of the request is not allowed to verify the
        identity.
    """
        IDENTITY_EVAL_STATE_UNSPECIFIED = 0
        MATCH = 1
        NOT_MATCH = 2
        NOT_SUPPORTED = 3
        INFO_DENIED = 4

    class IdentityStateValueValuesEnum(_messages.Enum):
        """Output only. The current state of an identity set in policy.

    Values:
      IDENTITY_STATE_UNSPECIFIED: Not used
      ACTIVE: Identity is active.
      DELETED: Identity is deleted.
    """
        IDENTITY_STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    identityEvalState = _messages.EnumField('IdentityEvalStateValueValuesEnum', 1)
    identityState = _messages.EnumField('IdentityStateValueValuesEnum', 2)