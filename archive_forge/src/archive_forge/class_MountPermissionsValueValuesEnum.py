from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MountPermissionsValueValuesEnum(_messages.Enum):
    """Mount permissions.

    Values:
      MOUNT_PERMISSIONS_UNSPECIFIED: Permissions were not specified.
      READ: NFS share can be mount with read-only permissions.
      READ_WRITE: NFS share can be mount with read-write permissions.
    """
    MOUNT_PERMISSIONS_UNSPECIFIED = 0
    READ = 1
    READ_WRITE = 2