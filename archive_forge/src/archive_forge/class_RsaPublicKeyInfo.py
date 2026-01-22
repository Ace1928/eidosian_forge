from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RsaPublicKeyInfo(_messages.Message):
    """Information of a RSA public key.

  Fields:
    keySize: Key size in bits (size of the modulus).
  """
    keySize = _messages.IntegerField(1, variant=_messages.Variant.INT32)