from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OidcConfig(_messages.Message):
    """Configuration for OIDC Auth flow.

  Fields:
    certificateAuthorityData: PEM-encoded CA for OIDC provider.
    clientId: ID for OIDC client application.
    clientSecret: Input only. Unencrypted OIDC client secret will be passed to
      the GKE Hub CLH.
    deployCloudConsoleProxy: Flag to denote if reverse proxy is used to
      connect to auth provider. This flag should be set to true when provider
      is not reachable by Google Cloud Console.
    enableAccessToken: Enable access token.
    encryptedClientSecret: Output only. Encrypted OIDC Client secret
    extraParams: Comma-separated list of key-value pairs.
    groupPrefix: Prefix to prepend to group name.
    groupsClaim: Claim in OIDC ID token that holds group information.
    issuerUri: URI for the OIDC provider. This should point to the level below
      .well-known/openid-configuration.
    kubectlRedirectUri: Registered redirect uri to redirect users going
      through OAuth flow using kubectl plugin.
    scopes: Comma-separated list of identifiers.
    userClaim: Claim in OIDC ID token that holds username.
    userPrefix: Prefix to prepend to user name.
  """
    certificateAuthorityData = _messages.StringField(1)
    clientId = _messages.StringField(2)
    clientSecret = _messages.StringField(3)
    deployCloudConsoleProxy = _messages.BooleanField(4)
    enableAccessToken = _messages.BooleanField(5)
    encryptedClientSecret = _messages.BytesField(6)
    extraParams = _messages.StringField(7)
    groupPrefix = _messages.StringField(8)
    groupsClaim = _messages.StringField(9)
    issuerUri = _messages.StringField(10)
    kubectlRedirectUri = _messages.StringField(11)
    scopes = _messages.StringField(12)
    userClaim = _messages.StringField(13)
    userPrefix = _messages.StringField(14)