from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityStyleValueValuesEnum(_messages.Enum):
    """Optional. Security Style of the Volume

    Values:
      SECURITY_STYLE_UNSPECIFIED: SecurityStyle is unspecified
      NTFS: SecurityStyle uses NTFS
      UNIX: SecurityStyle uses NTFS
    """
    SECURITY_STYLE_UNSPECIFIED = 0
    NTFS = 1
    UNIX = 2