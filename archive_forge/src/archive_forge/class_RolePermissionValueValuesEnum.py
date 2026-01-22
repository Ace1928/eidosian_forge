from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolePermissionValueValuesEnum(_messages.Enum):
    """Indicates whether the role granted by this binding contains the
    specified permission.

    Values:
      ROLE_PERMISSION_UNSPECIFIED: Default value. This value is unused.
      ROLE_PERMISSION_INCLUDED: The permission is included in the role.
      ROLE_PERMISSION_NOT_INCLUDED: The permission is not included in the
        role.
      ROLE_PERMISSION_UNKNOWN_INFO_DENIED: The user who created the Replay is
        not allowed to access the binding.
    """
    ROLE_PERMISSION_UNSPECIFIED = 0
    ROLE_PERMISSION_INCLUDED = 1
    ROLE_PERMISSION_NOT_INCLUDED = 2
    ROLE_PERMISSION_UNKNOWN_INFO_DENIED = 3