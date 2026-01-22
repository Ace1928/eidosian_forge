from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1OrgPolicyOverlayPolicyOverlay(_messages.Message):
    """A change to an OrgPolicy.

  Fields:
    policy: Optional. The new or updated OrgPolicy.
    policyParent: Optional. The parent of the policy we are attaching to.
      Example: "projects/123456"
  """
    policy = _messages.MessageField('GoogleCloudOrgpolicyV2Policy', 1)
    policyParent = _messages.StringField(2)