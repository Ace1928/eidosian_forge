from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyOrganizationsCustomConstraintsCreateRequest(_messages.Message):
    """A OrgpolicyOrganizationsCustomConstraintsCreateRequest object.

  Fields:
    googleCloudOrgpolicyV2CustomConstraint: A
      GoogleCloudOrgpolicyV2CustomConstraint resource to be passed as the
      request body.
    parent: Required. Must be in the following form: *
      `organizations/{organization_id}`
  """
    googleCloudOrgpolicyV2CustomConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2CustomConstraint', 1)
    parent = _messages.StringField(2, required=True)