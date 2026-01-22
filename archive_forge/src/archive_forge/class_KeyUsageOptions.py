from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyUsageOptions(_messages.Message):
    """KeyUsage.KeyUsageOptions corresponds to the key usage values described
  in https://tools.ietf.org/html/rfc5280#section-4.2.1.3.

  Fields:
    certSign: The key may be used to sign certificates.
    contentCommitment: The key may be used for cryptographic commitments. Note
      that this may also be referred to as "non-repudiation".
    crlSign: The key may be used sign certificate revocation lists.
    dataEncipherment: The key may be used to encipher data.
    decipherOnly: The key may be used to decipher only.
    digitalSignature: The key may be used for digital signatures.
    encipherOnly: The key may be used to encipher only.
    keyAgreement: The key may be used in a key agreement protocol.
    keyEncipherment: The key may be used to encipher other keys.
  """
    certSign = _messages.BooleanField(1)
    contentCommitment = _messages.BooleanField(2)
    crlSign = _messages.BooleanField(3)
    dataEncipherment = _messages.BooleanField(4)
    decipherOnly = _messages.BooleanField(5)
    digitalSignature = _messages.BooleanField(6)
    encipherOnly = _messages.BooleanField(7)
    keyAgreement = _messages.BooleanField(8)
    keyEncipherment = _messages.BooleanField(9)