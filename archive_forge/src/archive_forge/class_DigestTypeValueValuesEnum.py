from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DigestTypeValueValuesEnum(_messages.Enum):
    """The hash function used to generate the digest of the referenced
    DNSKEY.

    Values:
      DIGEST_TYPE_UNSPECIFIED: The DigestType is unspecified.
      SHA1: SHA-1. Not recommended for new deployments.
      SHA256: SHA-256.
      GOST3411: GOST R 34.11-94.
      SHA384: SHA-384.
    """
    DIGEST_TYPE_UNSPECIFIED = 0
    SHA1 = 1
    SHA256 = 2
    GOST3411 = 3
    SHA384 = 4