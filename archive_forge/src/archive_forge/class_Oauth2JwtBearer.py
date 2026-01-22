from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Oauth2JwtBearer(_messages.Message):
    """Parameters to support JSON Web Token (JWT) Profile for Oauth 2.0
  Authorization Grant based authentication. See
  https://tools.ietf.org/html/rfc7523 for more details.

  Fields:
    clientKey: Secret version reference containing a PKCS#8 PEM-encoded
      private key associated with the Client Certificate. This private key
      will be used to sign JWTs used for the jwt-bearer authorization grant.
      Specified in the form as: `projects/*/secrets/*/versions/*`.
    jwtClaims: JwtClaims providers fields to generate the token.
  """
    clientKey = _messages.MessageField('Secret', 1)
    jwtClaims = _messages.MessageField('JwtClaims', 2)