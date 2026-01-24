from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1OrgPolicyOverlayCustomConstraintOverlay(_messages.Message):
    """A change to an OrgPolicy custom constraint.

  Fields:
    customConstraint: Optional. The new or updated custom constraint.
    customConstraintParent: Optional. Resource the constraint is attached to.
      Example: "organization/989284"
  """
    customConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2CustomConstraint', 1)
    customConstraintParent = _messages.StringField(2)