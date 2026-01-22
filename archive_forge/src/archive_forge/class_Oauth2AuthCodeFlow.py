from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Oauth2AuthCodeFlow(_messages.Message):
    """Parameters to support Oauth 2.0 Auth Code Grant Authentication. See
  https://www.rfc-editor.org/rfc/rfc6749#section-1.3.1 for more details.

  Fields:
    authCode: Authorization code to be exchanged for access and refresh
      tokens.
    authUri: Auth URL for Authorization Code Flow
    clientId: Client ID for user-provided OAuth app.
    clientSecret: Client secret for user-provided OAuth app.
    enablePkce: Whether to enable PKCE when the user performs the auth code
      flow.
    pkceVerifier: PKCE verifier to be used during the auth code exchange.
    redirectUri: Redirect URI to be provided during the auth code exchange.
    scopes: Scopes the connection will request when the user performs the auth
      code flow.
  """
    authCode = _messages.StringField(1)
    authUri = _messages.StringField(2)
    clientId = _messages.StringField(3)
    clientSecret = _messages.MessageField('Secret', 4)
    enablePkce = _messages.BooleanField(5)
    pkceVerifier = _messages.StringField(6)
    redirectUri = _messages.StringField(7)
    scopes = _messages.StringField(8, repeated=True)