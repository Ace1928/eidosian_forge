from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1DevicePolicy(_messages.Message):
    """`DevicePolicy` specifies device specific restrictions necessary to
  acquire a given access level. A `DevicePolicy` specifies requirements for
  requests from devices to be granted access levels, it does not do any
  enforcement on the device. `DevicePolicy` acts as an AND over all specified
  fields, and each repeated field is an OR over its elements. Any unset fields
  are ignored. For example, if the proto is { os_type : DESKTOP_WINDOWS,
  os_type : DESKTOP_LINUX, encryption_status: ENCRYPTED}, then the
  DevicePolicy will be true for requests originating from encrypted Linux
  desktops and encrypted Windows desktops.

  Enums:
    AllowedDeviceManagementLevelsValueListEntryValuesEnum:
    AllowedEncryptionStatusesValueListEntryValuesEnum:

  Fields:
    allowedDeviceManagementLevels: Allowed device management levels, an empty
      list allows all management levels.
    allowedEncryptionStatuses: Allowed encryptions statuses, an empty list
      allows all statuses.
    osConstraints: Allowed OS versions, an empty list allows all types and all
      versions.
    requireAdminApproval: Whether the device needs to be approved by the
      customer admin.
    requireCorpOwned: Whether the device needs to be corp owned.
    requireScreenlock: Whether or not screenlock is required for the
      DevicePolicy to be true. Defaults to `false`.
  """

    class AllowedDeviceManagementLevelsValueListEntryValuesEnum(_messages.Enum):
        """AllowedDeviceManagementLevelsValueListEntryValuesEnum enum type.

    Values:
      MANAGEMENT_UNSPECIFIED: The device's management level is not specified
        or not known.
      NONE: The device is not managed.
      BASIC: Basic management is enabled, which is generally limited to
        monitoring and wiping the corporate account.
      COMPLETE: Complete device management. This includes more thorough
        monitoring and the ability to directly manage the device (such as
        remote wiping). This can be enabled through the Android Enterprise
        Platform.
    """
        MANAGEMENT_UNSPECIFIED = 0
        NONE = 1
        BASIC = 2
        COMPLETE = 3

    class AllowedEncryptionStatusesValueListEntryValuesEnum(_messages.Enum):
        """AllowedEncryptionStatusesValueListEntryValuesEnum enum type.

    Values:
      ENCRYPTION_UNSPECIFIED: The encryption status of the device is not
        specified or not known.
      ENCRYPTION_UNSUPPORTED: The device does not support encryption.
      UNENCRYPTED: The device supports encryption, but is currently
        unencrypted.
      ENCRYPTED: The device is encrypted.
    """
        ENCRYPTION_UNSPECIFIED = 0
        ENCRYPTION_UNSUPPORTED = 1
        UNENCRYPTED = 2
        ENCRYPTED = 3
    allowedDeviceManagementLevels = _messages.EnumField('AllowedDeviceManagementLevelsValueListEntryValuesEnum', 1, repeated=True)
    allowedEncryptionStatuses = _messages.EnumField('AllowedEncryptionStatusesValueListEntryValuesEnum', 2, repeated=True)
    osConstraints = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1OsConstraint', 3, repeated=True)
    requireAdminApproval = _messages.BooleanField(4)
    requireCorpOwned = _messages.BooleanField(5)
    requireScreenlock = _messages.BooleanField(6)