from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsKeysCreateRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsKeysCreateRequest object.

  Fields:
    googleCloudApigeeV1DeveloperAppKey: A GoogleCloudApigeeV1DeveloperAppKey
      resource to be passed as the request body.
    parent: Parent of the developer app key. Use the following structure in
      your request:
      'organizations/{org}/developers/{developerEmail}/apps/{appName}'
  """
    googleCloudApigeeV1DeveloperAppKey = _messages.MessageField('GoogleCloudApigeeV1DeveloperAppKey', 1)
    parent = _messages.StringField(2, required=True)