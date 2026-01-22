from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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