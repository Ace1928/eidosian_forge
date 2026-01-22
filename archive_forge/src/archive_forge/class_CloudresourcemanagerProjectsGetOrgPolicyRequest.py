from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsGetOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerProjectsGetOrgPolicyRequest object.

  Fields:
    getOrgPolicyRequest: A GetOrgPolicyRequest resource to be passed as the
      request body.
    projectsId: Part of `resource`. Name of the resource the `Policy` is set
      on.
  """
    getOrgPolicyRequest = _messages.MessageField('GetOrgPolicyRequest', 1)
    projectsId = _messages.StringField(2, required=True)