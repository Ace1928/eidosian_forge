from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Oauth2ClientCredentials(_messages.Message):
    """Parameters to support Oauth 2.0 Client Credentials Grant Authentication.
  See https://tools.ietf.org/html/rfc6749#section-1.3.4 for more details.

  Fields:
    clientId: The client identifier.
    clientSecret: Secret version reference containing the client secret.
  """
    clientId = _messages.StringField(1)
    clientSecret = _messages.MessageField('Secret', 2)