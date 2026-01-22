from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BakTypeValueValuesEnum(_messages.Enum):
    """Type of the bak content, FULL or DIFF.

      Values:
        BAK_TYPE_UNSPECIFIED: Default type.
        FULL: Full backup.
        DIFF: Differential backup.
        TLOG: SQL Server Transaction Log
      """
    BAK_TYPE_UNSPECIFIED = 0
    FULL = 1
    DIFF = 2
    TLOG = 3