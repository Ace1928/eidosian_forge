from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersClearOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerFoldersClearOrgPolicyRequest object.

  Fields:
    clearOrgPolicyRequest: A ClearOrgPolicyRequest resource to be passed as
      the request body.
    foldersId: Part of `resource`. Name of the resource for the `Policy` to
      clear.
  """
    clearOrgPolicyRequest = _messages.MessageField('ClearOrgPolicyRequest', 1)
    foldersId = _messages.StringField(2, required=True)