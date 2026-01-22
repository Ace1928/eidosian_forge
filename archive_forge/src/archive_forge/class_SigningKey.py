from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SigningKey(_messages.Message):
    """This defines the format used to record keys used in the software supply
  chain. An in-toto link is attested using one or more keys defined in the in-
  toto layout. An example of this is: { "key_id":
  "776a00e29f3559e0141b3b096f696abc6cfb0c657ab40f441132b345b0...", "key_type":
  "rsa", "public_key_value": "-----BEGIN PUBLIC
  KEY-----\\nMIIBojANBgkqhkiG9w0B...", "key_scheme": "rsassa-pss-sha256" } The
  format for in-toto's key definition can be found in section 4.2 of the in-
  toto specification.

  Fields:
    keyId: key_id is an identifier for the signing key.
    keyScheme: This field contains the corresponding signature scheme. Eg:
      "rsassa-pss-sha256".
    keyType: This field identifies the specific signing method. Eg: "rsa",
      "ed25519", and "ecdsa".
    publicKeyValue: This field contains the actual public key.
  """
    keyId = _messages.StringField(1)
    keyScheme = _messages.StringField(2)
    keyType = _messages.StringField(3)
    publicKeyValue = _messages.StringField(4)