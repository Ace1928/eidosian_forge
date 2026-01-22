from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IDPReferenceOIDC(_messages.Message):
    """Represents a reference to an OIDC provider.

  Fields:
    audience: Optional. The acceptable audience. Default is the unique_id of
      the Service Account.
    maxTokenLifetimeSeconds: This optional field allows enforcing a maximum
      lifetime for tokens. Using a lifetime that is as short as possible
      improves security since it prevents use of exfiltrated tokens after a
      certain amount of time. All tokens must specify both exp and iat or they
      will be rejected. If "nbf" is present we will reject tokens that are not
      yet valid. Expiration and lifetime will be enforced in the following
      way: - "exp" > "current time" is always required (expired tokens are
      rejected) - "iat" < "current time" + 300 seconds is required (tokens
      from the future . are rejected although a small amount of clock skew is
      tolerated). - If max_token_lifetime_seconds is set: "exp" - "iat" <
      max_token_lifetime_seconds will be checked - The default is otherwise to
      accept a max_token_lifetime_seconds of 3600 (1 hour)
    oidcJwks: Optional. OIDC verification keys in JWKS format (RFC 7517). It
      contains a list of OIDC verification keys that can be used to verify
      OIDC JWTs. When OIDC verification key is provided, it will be directly
      used to verify the OIDC JWT asserted by the IDP.
    url: The OpenID Connect URL. To use this Identity Binding, JWT 'iss' field
      should match this field. When URL is set, public keys will be fetched
      from the provided URL for credentials verification unless `oidc_jwks`
      field is set.
  """
    audience = _messages.StringField(1)
    maxTokenLifetimeSeconds = _messages.IntegerField(2)
    oidcJwks = _messages.BytesField(3)
    url = _messages.StringField(4)