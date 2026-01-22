from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityAwareProxy(_messages.Message):
    """Identity-Aware Proxy

  Fields:
    enabled: Whether the serving infrastructure will authenticate and
      authorize all incoming requests.If true, the oauth2_client_id and
      oauth2_client_secret fields must be non-empty.
    oauth2ClientId: OAuth2 client ID to use for the authentication flow.
    oauth2ClientSecret: OAuth2 client secret to use for the authentication
      flow.For security reasons, this value cannot be retrieved via the API.
      Instead, the SHA-256 hash of the value is returned in the
      oauth2_client_secret_sha256 field.@InputOnly
    oauth2ClientSecretSha256: Output only. Hex-encoded SHA-256 hash of the
      client secret.@OutputOnly
  """
    enabled = _messages.BooleanField(1)
    oauth2ClientId = _messages.StringField(2)
    oauth2ClientSecret = _messages.StringField(3)
    oauth2ClientSecretSha256 = _messages.StringField(4)