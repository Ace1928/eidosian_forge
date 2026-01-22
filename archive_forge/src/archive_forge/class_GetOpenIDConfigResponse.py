from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetOpenIDConfigResponse(_messages.Message):
    """GetOpenIDConfigResponse is an OIDC discovery document for the cluster.
  See the OpenID Connect Discovery 1.0 specification for details.

  Fields:
    cacheHeader: OnePlatform automatically extracts this field and uses it to
      set the HTTP Cache-Control header.
    claims_supported: Supported claims.
    grant_types: Supported grant types.
    id_token_signing_alg_values_supported: supported ID Token signing
      Algorithms.
    issuer: OIDC Issuer.
    jwks_uri: JSON Web Key uri.
    response_types_supported: Supported response types.
    subject_types_supported: Supported subject types.
  """
    cacheHeader = _messages.MessageField('HttpCacheControlResponseHeader', 1)
    claims_supported = _messages.StringField(2, repeated=True)
    grant_types = _messages.StringField(3, repeated=True)
    id_token_signing_alg_values_supported = _messages.StringField(4, repeated=True)
    issuer = _messages.StringField(5)
    jwks_uri = _messages.StringField(6)
    response_types_supported = _messages.StringField(7, repeated=True)
    subject_types_supported = _messages.StringField(8, repeated=True)