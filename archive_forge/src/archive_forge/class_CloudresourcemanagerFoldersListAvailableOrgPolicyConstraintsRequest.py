from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersListAvailableOrgPolicyConstraintsRequest(_messages.Message):
    """A CloudresourcemanagerFoldersListAvailableOrgPolicyConstraintsRequest
  object.

  Fields:
    foldersId: Part of `resource`. Name of the resource to list `Constraints`
      for.
    listAvailableOrgPolicyConstraintsRequest: A
      ListAvailableOrgPolicyConstraintsRequest resource to be passed as the
      request body.
  """
    foldersId = _messages.StringField(1, required=True)
    listAvailableOrgPolicyConstraintsRequest = _messages.MessageField('ListAvailableOrgPolicyConstraintsRequest', 2)