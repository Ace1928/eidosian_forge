from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedKeyType(_messages.Message):
    """Describes a "type" of key that may be used in a Certificate issued from
  a CaPool. Note that a single AllowedKeyType may refer to either a fully-
  qualified key algorithm, such as RSA 4096, or a family of key algorithms,
  such as any RSA key.

  Fields:
    ellipticCurve: Represents an allowed Elliptic Curve key type.
    rsa: Represents an allowed RSA key type.
  """
    ellipticCurve = _messages.MessageField('EcKeyType', 1)
    rsa = _messages.MessageField('RsaKeyType', 2)