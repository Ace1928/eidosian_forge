from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CDNPolicySignedTokenOptions(_messages.Message):
    """The configuration options for signed tokens.

  Enums:
    AllowedSignatureAlgorithmsValueListEntryValuesEnum:

  Fields:
    allowedSignatureAlgorithms: Optional. The allowed signature algorithms to
      use. Defaults to using only ED25519. You can specify up to 3 signature
      algorithms to use.
    tokenQueryParameter: Optional. The query parameter in which to find the
      token. The name must be 1-64 characters long and match the regular
      expression `[a-zA-Z]([a-zA-Z0-9_-])*` which means the first character
      must be a letter, and all following characters must be a dash,
      underscore, letter or digit. Defaults to `edge-cache-token`.
  """

    class AllowedSignatureAlgorithmsValueListEntryValuesEnum(_messages.Enum):
        """AllowedSignatureAlgorithmsValueListEntryValuesEnum enum type.

    Values:
      SIGNATURE_ALGORITHM_UNSPECIFIED: It is an error to specify
        ALGORITHM_UNSPECIFIED.
      ED25519: Use an Ed25519 signature scheme. The signature must be
        specified in the signature field of the token.
      HMAC_SHA_256: Use an HMAC based on a SHA-256 hash. The HMAC must be
        specified in the hmac field of the token.
      HMAC_SHA1: Use an HMAC based on a SHA1 hash. The HMAC must be specified
        in the hmac field of the token.
    """
        SIGNATURE_ALGORITHM_UNSPECIFIED = 0
        ED25519 = 1
        HMAC_SHA_256 = 2
        HMAC_SHA1 = 3
    allowedSignatureAlgorithms = _messages.EnumField('AllowedSignatureAlgorithmsValueListEntryValuesEnum', 1, repeated=True)
    tokenQueryParameter = _messages.StringField(2)