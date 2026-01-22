from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReauthSettings(_messages.Message):
    """Configuration for IAP reauthentication policies.

  Enums:
    MethodValueValuesEnum: Reauth method requested.
    PolicyTypeValueValuesEnum: How IAP determines the effective policy in
      cases of hierarchical policies. Policies are merged from higher in the
      hierarchy to lower in the hierarchy.

  Fields:
    maxAge: Reauth session lifetime, how long before a user has to
      reauthenticate again.
    method: Reauth method requested.
    policyType: How IAP determines the effective policy in cases of
      hierarchical policies. Policies are merged from higher in the hierarchy
      to lower in the hierarchy.
  """

    class MethodValueValuesEnum(_messages.Enum):
        """Reauth method requested.

    Values:
      METHOD_UNSPECIFIED: Reauthentication disabled.
      LOGIN: Prompts the user to log in again.
      PASSWORD: <no description>
      SECURE_KEY: User must use their secure key 2nd factor device.
      ENROLLED_SECOND_FACTORS: User can use any enabled 2nd factor.
    """
        METHOD_UNSPECIFIED = 0
        LOGIN = 1
        PASSWORD = 2
        SECURE_KEY = 3
        ENROLLED_SECOND_FACTORS = 4

    class PolicyTypeValueValuesEnum(_messages.Enum):
        """How IAP determines the effective policy in cases of hierarchical
    policies. Policies are merged from higher in the hierarchy to lower in the
    hierarchy.

    Values:
      POLICY_TYPE_UNSPECIFIED: Default value. This value is unused.
      MINIMUM: This policy acts as a minimum to other policies, lower in the
        hierarchy. Effective policy may only be the same or stricter.
      DEFAULT: This policy acts as a default if no other reauth policy is set.
    """
        POLICY_TYPE_UNSPECIFIED = 0
        MINIMUM = 1
        DEFAULT = 2
    maxAge = _messages.StringField(1)
    method = _messages.EnumField('MethodValueValuesEnum', 2)
    policyType = _messages.EnumField('PolicyTypeValueValuesEnum', 3)