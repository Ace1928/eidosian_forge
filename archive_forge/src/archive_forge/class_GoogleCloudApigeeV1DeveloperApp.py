from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DeveloperApp(_messages.Message):
    """A GoogleCloudApigeeV1DeveloperApp object.

  Fields:
    apiProducts: List of API products associated with the developer app.
    appFamily: Developer app family.
    appId: ID of the developer app.
    attributes: List of attributes for the developer app.
    callbackUrl: Callback URL used by OAuth 2.0 authorization servers to
      communicate authorization codes back to developer apps.
    createdAt: Output only. Time the developer app was created in milliseconds
      since epoch.
    credentials: Output only. Set of credentials for the developer app
      consisting of the consumer key/secret pairs associated with the API
      products.
    developerId: ID of the developer.
    keyExpiresIn: Expiration time, in milliseconds, for the consumer key that
      is generated for the developer app. If not set or left to the default
      value of `-1`, the API key never expires. The expiration time can't be
      updated after it is set.
    lastModifiedAt: Output only. Time the developer app was modified in
      milliseconds since epoch.
    name: Name of the developer app.
    scopes: Scopes to apply to the developer app. The specified scopes must
      already exist for the API product that you associate with the developer
      app.
    status: Status of the credential. Valid values include `approved` or
      `revoked`.
  """
    apiProducts = _messages.StringField(1, repeated=True)
    appFamily = _messages.StringField(2)
    appId = _messages.StringField(3)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 4, repeated=True)
    callbackUrl = _messages.StringField(5)
    createdAt = _messages.IntegerField(6)
    credentials = _messages.MessageField('GoogleCloudApigeeV1Credential', 7, repeated=True)
    developerId = _messages.StringField(8)
    keyExpiresIn = _messages.IntegerField(9)
    lastModifiedAt = _messages.IntegerField(10)
    name = _messages.StringField(11)
    scopes = _messages.StringField(12, repeated=True)
    status = _messages.StringField(13)