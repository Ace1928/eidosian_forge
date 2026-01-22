from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyFoldersPoliciesListRequest(_messages.Message):
    """A OrgpolicyFoldersPoliciesListRequest object.

  Fields:
    pageSize: Size of the pages to be returned. This is currently unsupported
      and will be ignored. The server may at any point start using this field
      to limit page size.
    pageToken: Page token used to retrieve the next page. This is currently
      unsupported and will be ignored. The server may at any point start using
      this field.
    parent: Required. The target Google Cloud resource that parents the set of
      constraints and policies that will be returned from this call. Must be
      in one of the following forms: * `projects/{project_number}` *
      `projects/{project_id}` * `folders/{folder_id}` *
      `organizations/{organization_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)