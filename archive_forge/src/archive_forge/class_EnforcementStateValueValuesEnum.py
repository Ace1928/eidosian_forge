from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnforcementStateValueValuesEnum(_messages.Enum):
    """Output only. Indicates whether the policy is enforced based on its
    version.

    Values:
      PAB_POLICY_ENFORCEMENT_STATE_UNSPECIFIED: An error occurred when
        checking whether a Principal Access Boundary policy is enforced based
        on its version.
      PAB_POLICY_ENFORCEMENT_STATE_ENFORCED: The Principal Access Boundary
        policy is enforced based on its version.
      PAB_POLICY_ENFORCEMENT_STATE_NOT_ENFORCED: The Principal Access Boundary
        policy is not enforced based on its version.
    """
    PAB_POLICY_ENFORCEMENT_STATE_UNSPECIFIED = 0
    PAB_POLICY_ENFORCEMENT_STATE_ENFORCED = 1
    PAB_POLICY_ENFORCEMENT_STATE_NOT_ENFORCED = 2