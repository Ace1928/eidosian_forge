from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetSyncAuthorizationRequest(_messages.Message):
    """A ApigeeOrganizationsGetSyncAuthorizationRequest object.

  Fields:
    googleCloudApigeeV1GetSyncAuthorizationRequest: A
      GoogleCloudApigeeV1GetSyncAuthorizationRequest resource to be passed as
      the request body.
    name: Required. Name of the Apigee organization. Use the following
      structure in your request: `organizations/{org}`
  """
    googleCloudApigeeV1GetSyncAuthorizationRequest = _messages.MessageField('GoogleCloudApigeeV1GetSyncAuthorizationRequest', 1)
    name = _messages.StringField(2, required=True)