from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudresourcemanagerOrganizationsUpdateRequest(_messages.Message):
    """A CloudresourcemanagerOrganizationsUpdateRequest object.

  Fields:
    organization: A Organization resource to be passed as the request body.
    organizationsId: Part of `name`. Output only. The resource name of the
      organization. This is the organization's relative path in the API. Its
      format is "organizations/[organization_id]". For example,
      "organizations/1234".
  """
    organization = _messages.MessageField('Organization', 1)
    organizationsId = _messages.StringField(2, required=True)