from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsGenerateKeyPairOrUpdateDeveloperAppStatusRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsGenerateKeyPairOrUpdateDeveloperAppSt
  atusRequest object.

  Fields:
    action: Action. Valid values are `approve` or `revoke`.
    googleCloudApigeeV1DeveloperApp: A GoogleCloudApigeeV1DeveloperApp
      resource to be passed as the request body.
    name: Required. Name of the developer app. Use the following structure in
      your request:
      `organizations/{org}/developers/{developer_email}/apps/{app}`
  """
    action = _messages.StringField(1)
    googleCloudApigeeV1DeveloperApp = _messages.MessageField('GoogleCloudApigeeV1DeveloperApp', 2)
    name = _messages.StringField(3, required=True)