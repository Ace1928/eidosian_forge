from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ArchiveTypeValueValuesEnum(_messages.Enum):
    """Type of archive files in this repository. The default behavior is DEB.

    Values:
      ARCHIVE_TYPE_UNSPECIFIED: Unspecified.
      DEB: DEB indicates that the archive contains binary files.
      DEB_SRC: DEB_SRC indicates that the archive contains source files.
    """
    ARCHIVE_TYPE_UNSPECIFIED = 0
    DEB = 1
    DEB_SRC = 2