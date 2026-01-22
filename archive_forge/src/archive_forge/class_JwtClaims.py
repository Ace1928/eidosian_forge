from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JwtClaims(_messages.Message):
    """JWT claims used for the jwt-bearer authorization grant.

  Fields:
    audience: Value for the "aud" claim.
    issuer: Value for the "iss" claim.
    subject: Value for the "sub" claim.
  """
    audience = _messages.StringField(1)
    issuer = _messages.StringField(2)
    subject = _messages.StringField(3)