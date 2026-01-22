from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AppGroupAppKey(_messages.Message):
    """AppGroupAppKey contains all the information associated with the
  credentials.

  Fields:
    apiProducts: Output only. List of API products and its status for which
      the credential can be used. **Note**: Use
      UpdateAppGroupAppKeyApiProductRequest API to make the association after
      the consumer key and secret are created.
    attributes: List of attributes associated with the credential.
    consumerKey: Immutable. Consumer key.
    consumerSecret: Secret key.
    expiresAt: Output only. Time the AppGroup app expires in milliseconds
      since epoch.
    expiresInSeconds: Immutable. Expiration time, in seconds, for the consumer
      key. If not set or left to the default value of `-1`, the API key never
      expires. The expiration time can't be updated after it is set.
    issuedAt: Output only. Time the AppGroup app was created in milliseconds
      since epoch.
    scopes: Scopes to apply to the app. The specified scope names must already
      be defined for the API product that you associate with the app.
    status: Status of the credential. Valid values include `approved` or
      `revoked`.
  """
    apiProducts = _messages.MessageField('GoogleCloudApigeeV1APIProductAssociation', 1, repeated=True)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    consumerKey = _messages.StringField(3)
    consumerSecret = _messages.StringField(4)
    expiresAt = _messages.IntegerField(5)
    expiresInSeconds = _messages.IntegerField(6)
    issuedAt = _messages.IntegerField(7)
    scopes = _messages.StringField(8, repeated=True)
    status = _messages.StringField(9)