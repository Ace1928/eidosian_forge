from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizeEnvironmentRequest(_messages.Message):
    """Request message for AuthorizeEnvironment.

  Fields:
    accessToken: The OAuth access token that should be sent to the
      environment.
    expireTime: The time when the credentials expire. If not set, defaults to
      one hour from when the server received the request.
    idToken: The OAuth ID token that should be sent to the environment.
  """
    accessToken = _messages.StringField(1)
    expireTime = _messages.StringField(2)
    idToken = _messages.StringField(3)