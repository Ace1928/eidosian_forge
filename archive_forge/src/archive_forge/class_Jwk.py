from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Jwk(_messages.Message):
    """Jwk is a JSON Web Key as specified in RFC 7517

  Fields:
    alg: Algorithm.
    crv: Used for ECDSA keys.
    e: Used for RSA keys.
    kid: Key ID.
    kty: Key Type.
    n: Used for RSA keys.
    use: Permitted uses for the public keys.
    x: Used for ECDSA keys.
    y: Used for ECDSA keys.
  """
    alg = _messages.StringField(1)
    crv = _messages.StringField(2)
    e = _messages.StringField(3)
    kid = _messages.StringField(4)
    kty = _messages.StringField(5)
    n = _messages.StringField(6)
    use = _messages.StringField(7)
    x = _messages.StringField(8)
    y = _messages.StringField(9)