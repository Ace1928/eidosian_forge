from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyBindingStateValueValuesEnum(_messages.Enum):
    """Output only. Indicates whether the policy binding takes effect.

    Values:
      POLICY_BINDING_STATE_UNSPECIFIED: An error occurred when checking
        whether the policy binding is enforced.
      POLICY_BINDING_STATE_ENFORCED: The policy binding is enforced.
      POLICY_BINDING_STATE_NOT_ENFORCED: The policy binding is not enforced.
    """
    POLICY_BINDING_STATE_UNSPECIFIED = 0
    POLICY_BINDING_STATE_ENFORCED = 1
    POLICY_BINDING_STATE_NOT_ENFORCED = 2