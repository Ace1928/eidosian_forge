from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerOrganizationsGetOrgPolicyRequest(_messages.Message):
    """A CloudresourcemanagerOrganizationsGetOrgPolicyRequest object.

  Fields:
    getOrgPolicyRequest: A GetOrgPolicyRequest resource to be passed as the
      request body.
    organizationsId: Part of `resource`. Name of the resource the `Policy` is
      set on.
  """
    getOrgPolicyRequest = _messages.MessageField('GetOrgPolicyRequest', 1)
    organizationsId = _messages.StringField(2, required=True)