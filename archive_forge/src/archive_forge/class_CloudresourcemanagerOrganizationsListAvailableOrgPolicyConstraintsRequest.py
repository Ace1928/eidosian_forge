from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerOrganizationsListAvailableOrgPolicyConstraintsRequest(_messages.Message):
    """A
  CloudresourcemanagerOrganizationsListAvailableOrgPolicyConstraintsRequest
  object.

  Fields:
    listAvailableOrgPolicyConstraintsRequest: A
      ListAvailableOrgPolicyConstraintsRequest resource to be passed as the
      request body.
    organizationsId: Part of `resource`. Name of the resource to list
      `Constraints` for.
  """
    listAvailableOrgPolicyConstraintsRequest = _messages.MessageField('ListAvailableOrgPolicyConstraintsRequest', 1)
    organizationsId = _messages.StringField(2, required=True)