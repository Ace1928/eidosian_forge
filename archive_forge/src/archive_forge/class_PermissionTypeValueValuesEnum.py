from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PermissionTypeValueValuesEnum(_messages.Enum):
    """The type associated with the permission.

    Values:
      PERMISSION_TYPE_UNSPECIFIED: Default. Should not be used.
      ADMIN_READ: Permissions that gate reading resource configuration or
        metadata.
      ADMIN_WRITE: Permissions that gate modification of resource
        configuration or metadata.
      DATA_READ: Permissions that gate reading user-provided data.
      DATA_WRITE: Permissions that gate writing user-provided data.
    """
    PERMISSION_TYPE_UNSPECIFIED = 0
    ADMIN_READ = 1
    ADMIN_WRITE = 2
    DATA_READ = 3
    DATA_WRITE = 4