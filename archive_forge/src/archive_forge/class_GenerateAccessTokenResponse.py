from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateAccessTokenResponse(_messages.Message):
    """A GenerateAccessTokenResponse object.

  Fields:
    accessToken: The OAuth 2.0 access token.
    expireTime: Token expiration time. The expiration time is always set.
  """
    accessToken = _messages.StringField(1)
    expireTime = _messages.StringField(2)