from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerOrganizationsListOrgPoliciesRequest(_messages.Message):
    """A CloudresourcemanagerOrganizationsListOrgPoliciesRequest object.

  Fields:
    listOrgPoliciesRequest: A ListOrgPoliciesRequest resource to be passed as
      the request body.
    organizationsId: Part of `resource`. Name of the resource to list Policies
      for.
  """
    listOrgPoliciesRequest = _messages.MessageField('ListOrgPoliciesRequest', 1)
    organizationsId = _messages.StringField(2, required=True)