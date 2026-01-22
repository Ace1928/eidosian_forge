from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyFoldersPoliciesCreateRequest(_messages.Message):
    """A OrgpolicyFoldersPoliciesCreateRequest object.

  Fields:
    googleCloudOrgpolicyV2Policy: A GoogleCloudOrgpolicyV2Policy resource to
      be passed as the request body.
    parent: Required. The Google Cloud resource that will parent the new
      policy. Must be in one of the following forms: *
      `projects/{project_number}` * `projects/{project_id}` *
      `folders/{folder_id}` * `organizations/{organization_id}`
  """
    googleCloudOrgpolicyV2Policy = _messages.MessageField('GoogleCloudOrgpolicyV2Policy', 1)
    parent = _messages.StringField(2, required=True)