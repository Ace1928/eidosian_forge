from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GidValueValuesEnum(_messages.Enum):
    """Specifies how each file's POSIX group ID (GID) attribute should be
    handled by the transfer. By default, GID is not preserved. Only applicable
    to transfers involving POSIX file systems, and ignored for other
    transfers.

    Values:
      GID_UNSPECIFIED: GID behavior is unspecified.
      GID_SKIP: Do not preserve GID during a transfer job.
      GID_NUMBER: Preserve GID during a transfer job.
    """
    GID_UNSPECIFIED = 0
    GID_SKIP = 1
    GID_NUMBER = 2