from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalSetPolicyRequest(_messages.Message):
    """Request message for `SetPolicy` method.

  Fields:
    disableNotification: Optional. Set the field as `true` to disable the
      onboarding notification.
    policy: Required. The policy to be applied to the `resource`.
    resource: Required. The resource for which the policy is being specified.
      This policy replaces any existing policy.
  """
    disableNotification = _messages.BooleanField(1)
    policy = _messages.MessageField('SasPortalPolicy', 2)
    resource = _messages.StringField(3)