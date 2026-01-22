from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsListAvailableOrgPolicyConstraintsRequest(_messages.Message):
    """A CloudresourcemanagerProjectsListAvailableOrgPolicyConstraintsRequest
  object.

  Fields:
    listAvailableOrgPolicyConstraintsRequest: A
      ListAvailableOrgPolicyConstraintsRequest resource to be passed as the
      request body.
    projectsId: Part of `resource`. Name of the resource to list `Constraints`
      for.
  """
    listAvailableOrgPolicyConstraintsRequest = _messages.MessageField('ListAvailableOrgPolicyConstraintsRequest', 1)
    projectsId = _messages.StringField(2, required=True)