from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnSourceDiskDeleteValueValuesEnum(_messages.Enum):
    """Specifies the behavior to apply to scheduled snapshots when the source
    disk is deleted.

    Values:
      APPLY_RETENTION_POLICY: <no description>
      KEEP_AUTO_SNAPSHOTS: <no description>
      UNSPECIFIED_ON_SOURCE_DISK_DELETE: <no description>
    """
    APPLY_RETENTION_POLICY = 0
    KEEP_AUTO_SNAPSHOTS = 1
    UNSPECIFIED_ON_SOURCE_DISK_DELETE = 2