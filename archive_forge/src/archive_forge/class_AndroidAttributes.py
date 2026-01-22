from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AndroidAttributes(_messages.Message):
    """Resource representing the Android specific attributes of a Device.

  Enums:
    OwnershipPrivilegeValueValuesEnum: Ownership privileges on device.

  Fields:
    ctsProfileMatch: Whether the device passes Android CTS compliance.
    enabledUnknownSources: Whether applications from unknown sources can be
      installed on device.
    hasPotentiallyHarmfulApps: Whether any potentially harmful apps were
      detected on the device.
    ownerProfileAccount: Whether this account is on an owner/primary profile.
      For phones, only true for owner profiles. Android 4+ devices can have
      secondary or restricted user profiles.
    ownershipPrivilege: Ownership privileges on device.
    supportsWorkProfile: Whether device supports Android work profiles. If
      false, this service will not block access to corp data even if an
      administrator turns on the "Enforce Work Profile" policy.
    verifiedBoot: Whether Android verified boot status is GREEN.
    verifyAppsEnabled: Whether Google Play Protect Verify Apps is enabled.
  """

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
    ctsProfileMatch = _messages.BooleanField(1)
    enabledUnknownSources = _messages.BooleanField(2)
    hasPotentiallyHarmfulApps = _messages.BooleanField(3)
    ownerProfileAccount = _messages.BooleanField(4)
    ownershipPrivilege = _messages.EnumField('OwnershipPrivilegeValueValuesEnum', 5)
    supportsWorkProfile = _messages.BooleanField(6)
    verifiedBoot = _messages.BooleanField(7)
    verifyAppsEnabled = _messages.BooleanField(8)