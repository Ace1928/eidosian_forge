from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerOrganizationsGetRequest(_messages.Message):
    """A CloudresourcemanagerOrganizationsGetRequest object.

  Fields:
    organizationsId: Part of `name`. The resource name of the Organization to
      fetch. This is the organization's relative path in the API, formatted as
      "organizations/[organizationId]". For example, "organizations/1234".
  """
    organizationsId = _messages.StringField(1, required=True)