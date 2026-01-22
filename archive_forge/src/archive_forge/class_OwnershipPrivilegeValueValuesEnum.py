from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OwnershipPrivilegeValueValuesEnum(_messages.Enum):
    """Ownership privileges on device.

    Values:
      OWNERSHIP_PRIVILEGE_UNSPECIFIED: Ownership privilege is not set.
      DEVICE_ADMINISTRATOR: Active device administrator privileges on the
        device.
      PROFILE_OWNER: Profile Owner privileges. The account is in a managed
        corporate profile.
      DEVICE_OWNER: Device Owner privileges on the device.
    """
    OWNERSHIP_PRIVILEGE_UNSPECIFIED = 0
    DEVICE_ADMINISTRATOR = 1
    PROFILE_OWNER = 2
    DEVICE_OWNER = 3