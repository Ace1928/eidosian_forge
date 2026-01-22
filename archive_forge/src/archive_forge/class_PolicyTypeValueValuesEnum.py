from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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