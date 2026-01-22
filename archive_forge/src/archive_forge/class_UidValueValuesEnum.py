from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UidValueValuesEnum(_messages.Enum):
    """Specifies how each file's POSIX user ID (UID) attribute should be
    handled by the transfer. By default, UID is not preserved. Only applicable
    to transfers involving POSIX file systems, and ignored for other
    transfers.

    Values:
      UID_UNSPECIFIED: UID behavior is unspecified.
      UID_SKIP: Do not preserve UID during a transfer job.
      UID_NUMBER: Preserve UID during a transfer job.
    """
    UID_UNSPECIFIED = 0
    UID_SKIP = 1
    UID_NUMBER = 2