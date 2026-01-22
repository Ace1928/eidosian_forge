from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceProvenanceHashValueListEntryValuesEnum(_messages.Enum):
    """SourceProvenanceHashValueListEntryValuesEnum enum type.

    Values:
      NONE: No hash requested.
      SHA256: Use a sha256 hash.
      MD5: Use a md5 hash.
      SHA512: Use a sha512 hash.
    """
    NONE = 0
    SHA256 = 1
    MD5 = 2
    SHA512 = 3