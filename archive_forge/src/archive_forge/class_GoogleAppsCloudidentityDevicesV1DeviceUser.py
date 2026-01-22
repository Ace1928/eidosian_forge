from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1DeviceUser(_messages.Message):
    """Represents a user's use of a Device in the Cloud Identity Devices API. A
  DeviceUser is a resource representing a user's use of a Device

  Enums:
    CompromisedStateValueValuesEnum: Compromised State of the DeviceUser
      object
    ManagementStateValueValuesEnum: Output only. Management state of the user
      on the device.
    PasswordStateValueValuesEnum: Password state of the DeviceUser object

  Fields:
    compromisedState: Compromised State of the DeviceUser object
    createTime: When the user first signed in to the device
    firstSyncTime: Output only. Most recent time when user registered with
      this service.
    languageCode: Output only. Default locale used on device, in IETF BCP-47
      format.
    lastSyncTime: Output only. Last time when user synced with policies.
    managementState: Output only. Management state of the user on the device.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      DeviceUser in format: `devices/{device}/deviceUsers/{device_user}`,
      where `device_user` uniquely identifies a user's use of a device.
    passwordState: Password state of the DeviceUser object
    userAgent: Output only. User agent on the device for this specific user
    userEmail: Email address of the user registered on the device.
  """

    class CompromisedStateValueValuesEnum(_messages.Enum):
        """Compromised State of the DeviceUser object

    Values:
      COMPROMISED_STATE_UNSPECIFIED: Compromised state of Device User account
        is unknown or unspecified.
      COMPROMISED: Device User Account is compromised.
      NOT_COMPROMISED: Device User Account is not compromised.
    """
        COMPROMISED_STATE_UNSPECIFIED = 0
        COMPROMISED = 1
        NOT_COMPROMISED = 2

    class ManagementStateValueValuesEnum(_messages.Enum):
        """Output only. Management state of the user on the device.

    Values:
      MANAGEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      WIPING: This user's data and profile is being removed from the device.
      WIPED: This user's data and profile is removed from the device.
      APPROVED: User is approved to access data on the device.
      BLOCKED: User is blocked from accessing data on the device.
      PENDING_APPROVAL: User is awaiting approval.
      UNENROLLED: User is unenrolled from Advanced Windows Management, but the
        Windows account is still intact.
    """
        MANAGEMENT_STATE_UNSPECIFIED = 0
        WIPING = 1
        WIPED = 2
        APPROVED = 3
        BLOCKED = 4
        PENDING_APPROVAL = 5
        UNENROLLED = 6

    class PasswordStateValueValuesEnum(_messages.Enum):
        """Password state of the DeviceUser object

    Values:
      PASSWORD_STATE_UNSPECIFIED: Password state not set.
      PASSWORD_SET: Password set in object.
      PASSWORD_NOT_SET: Password not set in object.
    """
        PASSWORD_STATE_UNSPECIFIED = 0
        PASSWORD_SET = 1
        PASSWORD_NOT_SET = 2
    compromisedState = _messages.EnumField('CompromisedStateValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    firstSyncTime = _messages.StringField(3)
    languageCode = _messages.StringField(4)
    lastSyncTime = _messages.StringField(5)
    managementState = _messages.EnumField('ManagementStateValueValuesEnum', 6)
    name = _messages.StringField(7)
    passwordState = _messages.EnumField('PasswordStateValueValuesEnum', 8)
    userAgent = _messages.StringField(9)
    userEmail = _messages.StringField(10)