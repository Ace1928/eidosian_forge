from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationLoggingOptions(_messages.Message):
    """Authorization-related information used by Cloud Audit Logging.

  Enums:
    PermissionTypeValueValuesEnum: The type of the permission that was
      checked.

  Fields:
    permissionType: The type of the permission that was checked.
  """

    class PermissionTypeValueValuesEnum(_messages.Enum):
        """The type of the permission that was checked.

    Values:
      PERMISSION_TYPE_UNSPECIFIED: Default. Should not be used.
      ADMIN_READ: A read of admin (meta) data.
      ADMIN_WRITE: A write of admin (meta) data.
      DATA_READ: A read of standard data.
      DATA_WRITE: A write of standard data.
    """
        PERMISSION_TYPE_UNSPECIFIED = 0
        ADMIN_READ = 1
        ADMIN_WRITE = 2
        DATA_READ = 3
        DATA_WRITE = 4
    permissionType = _messages.EnumField('PermissionTypeValueValuesEnum', 1)