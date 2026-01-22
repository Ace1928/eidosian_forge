from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerOrganizationsLocationsSetupServiceRequest(_messages.Message):
    """A PrivilegedaccessmanagerOrganizationsLocationsSetupServiceRequest
  object.

  Fields:
    parent: Required. The parent resource for which this service needs to be
      setup. Should be in one of the following formats: * `projects/{project-
      number|project-id}/locations/{region}` * `folders/{folder-
      number}/locations/{region}` * `organizations/{organization-
      number}/locations/{region}`
    setupServiceRequest: A SetupServiceRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    setupServiceRequest = _messages.MessageField('SetupServiceRequest', 2)