from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1betaWorkloadIdentityPoolProviderOidc(_messages.Message):
    """Represents an OpenId Connect 1.0 identity provider.

  Fields:
    allowedAudiences: Acceptable values for the `aud` field (audience) in the
      OIDC token. Token exchange requests are rejected if the token audience
      does not match one of the configured values. Each audience may be at
      most 256 characters. A maximum of 10 audiences may be configured. If
      this list is empty, the OIDC token audience must be equal to the full
      canonical resource name of the WorkloadIdentityPoolProvider, with or
      without the HTTPS prefix. For example: ``` //iam.googleapis.com/projects
      //locations//workloadIdentityPools//providers/ https://iam.googleapis.co
      m/projects//locations//workloadIdentityPools//providers/ ```
    issuerUri: Required. The OIDC issuer URL. Must be an HTTPS endpoint.
    jwksJson: Optional. OIDC JWKs in JSON String format. For details on
      definition of a JWK, see https://tools.ietf.org/html/rfc7517. If not
      set, then we use the `jwks_uri` from the discovery document fetched from
      the .well-known path for the `issuer_uri`. Currently, RSA and EC
      asymmetric keys are supported. The JWK must use following format and
      include only the following fields: { "keys": [ { "kty": "RSA/EC", "alg":
      "", "use": "sig", "kid": "", "n": "", "e": "", "x": "", "y": "", "crv":
      "" } ] }
  """
    allowedAudiences = _messages.StringField(1, repeated=True)
    issuerUri = _messages.StringField(2)
    jwksJson = _messages.StringField(3)