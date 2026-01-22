from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersGetOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerFoldersGetOrgPolicyRequest object.

  Fields:
    foldersId: Part of `resource`. Name of the resource the `Policy` is set
      on.
    getOrgPolicyRequest: A GetOrgPolicyRequest resource to be passed as the
      request body.
  """
    foldersId = _messages.StringField(1, required=True)
    getOrgPolicyRequest = _messages.MessageField('GetOrgPolicyRequest', 2)