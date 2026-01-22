from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowAccessStateValueValuesEnum(_messages.Enum):
    """Required. Indicates whether _this policy_ provides the specified
    permission to the specified principal for the specified resource. This
    field does _not_ indicate whether the principal actually has the
    permission for the resource. There might be another policy that overrides
    this policy. To determine whether the principal actually has the
    permission, use the `overall_access_state` field in the
    TroubleshootIamPolicyResponse.

    Values:
      ALLOW_ACCESS_STATE_UNSPECIFIED: Not specified.
      ALLOW_ACCESS_STATE_GRANTED: The allow policy gives the principal the
        permission.
      ALLOW_ACCESS_STATE_NOT_GRANTED: The allow policy doesn't give the
        principal the permission.
      ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL: The allow policy gives the
        principal the permission if a condition expression evaluate to `true`.
        However, the sender of the request didn't provide enough context for
        Policy Troubleshooter to evaluate the condition expression.
      ALLOW_ACCESS_STATE_UNKNOWN_INFO: The sender of the request doesn't have
        access to all of the allow policies that Policy Troubleshooter needs
        to evaluate the principal's access.
    """
    ALLOW_ACCESS_STATE_UNSPECIFIED = 0
    ALLOW_ACCESS_STATE_GRANTED = 1
    ALLOW_ACCESS_STATE_NOT_GRANTED = 2
    ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
    ALLOW_ACCESS_STATE_UNKNOWN_INFO = 4