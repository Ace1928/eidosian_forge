from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookGenericWebServiceOAuthConfig(_messages.Message):
    """Represents configuration of OAuth client credential flow for 3rd party
  API authentication.

  Fields:
    clientId: Required. The client ID provided by the 3rd party platform.
    clientSecret: Required. The client secret provided by the 3rd party
      platform.
    scopes: Optional. The OAuth scopes to grant.
    tokenEndpoint: Required. The token endpoint provided by the 3rd party
      platform to exchange an access token.
  """
    clientId = _messages.StringField(1)
    clientSecret = _messages.StringField(2)
    scopes = _messages.StringField(3, repeated=True)
    tokenEndpoint = _messages.StringField(4)