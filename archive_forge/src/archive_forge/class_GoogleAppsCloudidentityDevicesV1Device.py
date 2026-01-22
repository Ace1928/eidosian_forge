from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1Device(_messages.Message):
    """ A Device within the Cloud Identity Devices API. Represents a Device
  known to Google Cloud, independent of the device ownership, type, and
  whether it is assigned or in use by a user.

  Enums:
    CompromisedStateValueValuesEnum: Output only. Represents whether the
      Device is compromised.
    DeviceTypeValueValuesEnum: Output only. Type of device.
    EncryptionStateValueValuesEnum: Output only. Device encryption state.
    ManagementStateValueValuesEnum: Output only. Management state of the
      device
    OwnerTypeValueValuesEnum: Output only. Whether the device is owned by the
      company or an individual

  Fields:
    androidSpecificAttributes: Output only. Attributes specific to Android
      devices.
    assetTag: Asset tag of the device.
    basebandVersion: Output only. Baseband version of the device.
    bootloaderVersion: Output only. Device bootloader version. Example: 0.6.7.
    brand: Output only. Device brand. Example: Samsung.
    buildNumber: Output only. Build number of the device.
    compromisedState: Output only. Represents whether the Device is
      compromised.
    createTime: Output only. When the Company-Owned device was imported. This
      field is empty for BYOD devices.
    deviceId: Unique identifier for the device.
    deviceType: Output only. Type of device.
    enabledDeveloperOptions: Output only. Whether developer options is enabled
      on device.
    enabledUsbDebugging: Output only. Whether USB debugging is enabled on
      device.
    encryptionState: Output only. Device encryption state.
    endpointVerificationSpecificAttributes: Output only. Attributes specific
      to [Endpoint Verification](https://cloud.google.com/endpoint-
      verification/docs/overview) devices.
    hostname: Host name of the device.
    imei: Output only. IMEI number of device if GSM device; empty otherwise.
    kernelVersion: Output only. Kernel version of the device.
    lastSyncTime: Most recent time when device synced with this service.
    managementState: Output only. Management state of the device
    manufacturer: Output only. Device manufacturer. Example: Motorola.
    meid: Output only. MEID number of device if CDMA device; empty otherwise.
    model: Output only. Model name of device. Example: Pixel 3.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the Device
      in format: `devices/{device}`, where device is the unique id assigned to
      the Device.
    networkOperator: Output only. Mobile or network operator of device, if
      available.
    osVersion: Output only. OS version of the device. Example: Android 8.1.0.
    otherAccounts: Output only. Domain name for Google accounts on device.
      Type for other accounts on device. On Android, will only be populated if
      |ownership_privilege| is |PROFILE_OWNER| or |DEVICE_OWNER|. Does not
      include the account signed in to the device policy app if that account's
      domain has only one account. Examples: "com.example", "xyz.com".
    ownerType: Output only. Whether the device is owned by the company or an
      individual
    releaseVersion: Output only. OS release version. Example: 6.0.
    securityPatchTime: Output only. OS security patch update time on device.
    serialNumber: Serial Number of device. Example: HT82V1A01076.
    wifiMacAddresses: WiFi MAC addresses of device.
  """

    class CompromisedStateValueValuesEnum(_messages.Enum):
        """Output only. Represents whether the Device is compromised.

    Values:
      COMPROMISED_STATE_UNSPECIFIED: Default value.
      COMPROMISED: The device is compromised (currently, this means Android
        device is rooted).
      UNCOMPROMISED: The device is safe (currently, this means Android device
        is unrooted).
    """
        COMPROMISED_STATE_UNSPECIFIED = 0
        COMPROMISED = 1
        UNCOMPROMISED = 2

    class DeviceTypeValueValuesEnum(_messages.Enum):
        """Output only. Type of device.

    Values:
      DEVICE_TYPE_UNSPECIFIED: Unknown device type
      ANDROID: Device is an Android device
      IOS: Device is an iOS device
      GOOGLE_SYNC: Device is a Google Sync device.
      WINDOWS: Device is a Windows device.
      MAC_OS: Device is a MacOS device.
      LINUX: Device is a Linux device.
      CHROME_OS: Device is a ChromeOS device.
    """
        DEVICE_TYPE_UNSPECIFIED = 0
        ANDROID = 1
        IOS = 2
        GOOGLE_SYNC = 3
        WINDOWS = 4
        MAC_OS = 5
        LINUX = 6
        CHROME_OS = 7

    class EncryptionStateValueValuesEnum(_messages.Enum):
        """Output only. Device encryption state.

    Values:
      ENCRYPTION_STATE_UNSPECIFIED: Encryption Status is not set.
      UNSUPPORTED_BY_DEVICE: Device doesn't support encryption.
      ENCRYPTED: Device is encrypted.
      NOT_ENCRYPTED: Device is not encrypted.
    """
        ENCRYPTION_STATE_UNSPECIFIED = 0
        UNSUPPORTED_BY_DEVICE = 1
        ENCRYPTED = 2
        NOT_ENCRYPTED = 3

    class ManagementStateValueValuesEnum(_messages.Enum):
        """Output only. Management state of the device

    Values:
      MANAGEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      APPROVED: Device is approved.
      BLOCKED: Device is blocked.
      PENDING: Device is pending approval.
      UNPROVISIONED: The device is not provisioned. Device will start from
        this state until some action is taken (i.e. a user starts using the
        device).
      WIPING: Data and settings on the device are being removed.
      WIPED: All data and settings on the device are removed.
    """
        MANAGEMENT_STATE_UNSPECIFIED = 0
        APPROVED = 1
        BLOCKED = 2
        PENDING = 3
        UNPROVISIONED = 4
        WIPING = 5
        WIPED = 6

    class OwnerTypeValueValuesEnum(_messages.Enum):
        """Output only. Whether the device is owned by the company or an
    individual

    Values:
      DEVICE_OWNERSHIP_UNSPECIFIED: Default value. The value is unused.
      COMPANY: Company owns the device.
      BYOD: Bring Your Own Device (i.e. individual owns the device)
    """
        DEVICE_OWNERSHIP_UNSPECIFIED = 0
        COMPANY = 1
        BYOD = 2
    androidSpecificAttributes = _messages.MessageField('GoogleAppsCloudidentityDevicesV1AndroidAttributes', 1)
    assetTag = _messages.StringField(2)
    basebandVersion = _messages.StringField(3)
    bootloaderVersion = _messages.StringField(4)
    brand = _messages.StringField(5)
    buildNumber = _messages.StringField(6)
    compromisedState = _messages.EnumField('CompromisedStateValueValuesEnum', 7)
    createTime = _messages.StringField(8)
    deviceId = _messages.StringField(9)
    deviceType = _messages.EnumField('DeviceTypeValueValuesEnum', 10)
    enabledDeveloperOptions = _messages.BooleanField(11)
    enabledUsbDebugging = _messages.BooleanField(12)
    encryptionState = _messages.EnumField('EncryptionStateValueValuesEnum', 13)
    endpointVerificationSpecificAttributes = _messages.MessageField('GoogleAppsCloudidentityDevicesV1EndpointVerificationSpecificAttributes', 14)
    hostname = _messages.StringField(15)
    imei = _messages.StringField(16)
    kernelVersion = _messages.StringField(17)
    lastSyncTime = _messages.StringField(18)
    managementState = _messages.EnumField('ManagementStateValueValuesEnum', 19)
    manufacturer = _messages.StringField(20)
    meid = _messages.StringField(21)
    model = _messages.StringField(22)
    name = _messages.StringField(23)
    networkOperator = _messages.StringField(24)
    osVersion = _messages.StringField(25)
    otherAccounts = _messages.StringField(26, repeated=True)
    ownerType = _messages.EnumField('OwnerTypeValueValuesEnum', 27)
    releaseVersion = _messages.StringField(28)
    securityPatchTime = _messages.StringField(29)
    serialNumber = _messages.StringField(30)
    wifiMacAddresses = _messages.StringField(31, repeated=True)