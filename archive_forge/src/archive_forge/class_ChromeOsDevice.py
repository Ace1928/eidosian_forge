from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChromeOsDevice(_messages.Message):
    """JSON template for Chrome Os Device resource in Directory API.

  Messages:
    ActiveTimeRangesValueListEntry: A ActiveTimeRangesValueListEntry object.
    CpuStatusReportsValueListEntry: A CpuStatusReportsValueListEntry object.
    DeviceFilesValueListEntry: A DeviceFilesValueListEntry object.
    DiskVolumeReportsValueListEntry: A DiskVolumeReportsValueListEntry object.
    RecentUsersValueListEntry: A RecentUsersValueListEntry object.
    SystemRamFreeReportsValueListEntry: A SystemRamFreeReportsValueListEntry
      object.
    TpmVersionInfoValue: Trusted Platform Module (TPM) (Read-only)

  Fields:
    activeTimeRanges: List of active time ranges (Read-only)
    annotatedAssetId: AssetId specified during enrollment or through later
      annotation
    annotatedLocation: Address or location of the device as noted by the
      administrator
    annotatedUser: User of the device
    autoUpdateExpiration: (Read-only) The timestamp after which the device
      will stop receiving Chrome updates or support
    bootMode: Chromebook boot mode (Read-only)
    cpuStatusReports: Reports of CPU utilization and temperature (Read-only)
    deviceFiles: List of device files to download (Read-only)
    deviceId: Unique identifier of Chrome OS Device (Read-only)
    diskVolumeReports: Reports of disk space and other info about
      mounted/connected volumes.
    dockMacAddress: (Read-only) Built-in MAC address for the docking station
      that the device connected to. Factory sets Media access control address
      (MAC address) assigned for use by a dock. Currently this is only
      supported on the Dell Arcada / Sarien devices and the Dell WD19 / WD19TB
      Docking Station. It is reserved specifically for MAC pass through device
      policy. The format is twelve (12) hexadecimal digits without any
      delimiter (uppercase letters). This is only relevant for Dell devices.
    etag: ETag of the resource.
    ethernetMacAddress: Chromebook Mac Address on ethernet network interface
      (Read-only)
    ethernetMacAddress0: (Read-only) MAC address used by the Chromebook's
      internal ethernet port, and for onboard network (ethernet) interface.
      The format is twelve (12) hexadecimal digits without any delimiter
      (uppercase letters). This is only relevant for some devices.
    firmwareVersion: Chromebook firmware version (Read-only)
    kind: Kind of resource this is.
    lastEnrollmentTime: Date and time the device was last enrolled (Read-only)
    lastSync: Date and time the device was last synchronized with the policy
      settings in the G Suite administrator control panel (Read-only)
    macAddress: Chromebook Mac Address on wifi network interface (Read-only)
    manufactureDate: (Read-only) The date the device was manufactured in yyyy-
      mm-dd format.
    meid: Contains either the Mobile Equipment identifier (MEID) or the
      International Mobile Equipment Identity (IMEI) for the 3G mobile card in
      the Chromebook (Read-only)
    model: Chromebook Model (Read-only)
    notes: Notes added by the administrator
    orderNumber: Chromebook order number (Read-only)
    orgUnitPath: OrgUnit of the device
    osVersion: Chromebook Os Version (Read-only)
    platformVersion: Chromebook platform version (Read-only)
    recentUsers: List of recent device users, in descending order by last
      login time (Read-only)
    serialNumber: Chromebook serial number (Read-only)
    status: status of the device (Read-only)
    supportEndDate: Final date the device will be supported (Read-only)
    systemRamFreeReports: Reports of amounts of available RAM memory (Read-
      only)
    systemRamTotal: Total RAM on the device [in bytes] (Read-only)
    tpmVersionInfo: Trusted Platform Module (TPM) (Read-only)
    willAutoRenew: Will Chromebook auto renew after support end date (Read-
      only)
  """

    class ActiveTimeRangesValueListEntry(_messages.Message):
        """A ActiveTimeRangesValueListEntry object.

    Fields:
      activeTime: Duration in milliseconds
      date: Date of usage
    """
        activeTime = _messages.IntegerField(1, variant=_messages.Variant.INT32)
        date = extra_types.DateField(2)

    class CpuStatusReportsValueListEntry(_messages.Message):
        """A CpuStatusReportsValueListEntry object.

    Messages:
      CpuTemperatureInfoValueListEntry: A CpuTemperatureInfoValueListEntry
        object.

    Fields:
      cpuTemperatureInfo: List of CPU temperature samples.
      cpuUtilizationPercentageInfo: A integer attribute.
      reportTime: Date and time the report was received.
    """

        class CpuTemperatureInfoValueListEntry(_messages.Message):
            """A CpuTemperatureInfoValueListEntry object.

      Fields:
        label: CPU label
        temperature: Temperature in Celsius degrees.
      """
            label = _messages.StringField(1)
            temperature = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        cpuTemperatureInfo = _messages.MessageField('CpuTemperatureInfoValueListEntry', 1, repeated=True)
        cpuUtilizationPercentageInfo = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)
        reportTime = _message_types.DateTimeField(3)

    class DeviceFilesValueListEntry(_messages.Message):
        """A DeviceFilesValueListEntry object.

    Fields:
      createTime: Date and time the file was created
      downloadUrl: File download URL
      name: File name
      type: File type
    """
        createTime = _message_types.DateTimeField(1)
        downloadUrl = _messages.StringField(2)
        name = _messages.StringField(3)
        type = _messages.StringField(4)

    class DiskVolumeReportsValueListEntry(_messages.Message):
        """A DiskVolumeReportsValueListEntry object.

    Messages:
      VolumeInfoValueListEntry: A VolumeInfoValueListEntry object.

    Fields:
      volumeInfo: Disk volumes
    """

        class VolumeInfoValueListEntry(_messages.Message):
            """A VolumeInfoValueListEntry object.

      Fields:
        storageFree: Free disk space [in bytes]
        storageTotal: Total disk space [in bytes]
        volumeId: Volume id
      """
            storageFree = _messages.IntegerField(1)
            storageTotal = _messages.IntegerField(2)
            volumeId = _messages.StringField(3)
        volumeInfo = _messages.MessageField('VolumeInfoValueListEntry', 1, repeated=True)

    class RecentUsersValueListEntry(_messages.Message):
        """A RecentUsersValueListEntry object.

    Fields:
      email: Email address of the user. Present only if the user type is
        managed
      type: The type of the user
    """
        email = _messages.StringField(1)
        type = _messages.StringField(2)

    class SystemRamFreeReportsValueListEntry(_messages.Message):
        """A SystemRamFreeReportsValueListEntry object.

    Fields:
      reportTime: Date and time the report was received.
      systemRamFreeInfo: A string attribute.
    """
        reportTime = _message_types.DateTimeField(1)
        systemRamFreeInfo = _messages.IntegerField(2, repeated=True)

    class TpmVersionInfoValue(_messages.Message):
        """Trusted Platform Module (TPM) (Read-only)

    Fields:
      family: TPM family.
      firmwareVersion: TPM firmware version.
      manufacturer: TPM manufacturer code.
      specLevel: TPM specification level.
      tpmModel: TPM model number.
      vendorSpecific: Vendor-specific information such as Vendor ID.
    """
        family = _messages.StringField(1)
        firmwareVersion = _messages.StringField(2)
        manufacturer = _messages.StringField(3)
        specLevel = _messages.StringField(4)
        tpmModel = _messages.StringField(5)
        vendorSpecific = _messages.StringField(6)
    activeTimeRanges = _messages.MessageField('ActiveTimeRangesValueListEntry', 1, repeated=True)
    annotatedAssetId = _messages.StringField(2)
    annotatedLocation = _messages.StringField(3)
    annotatedUser = _messages.StringField(4)
    autoUpdateExpiration = _messages.IntegerField(5)
    bootMode = _messages.StringField(6)
    cpuStatusReports = _messages.MessageField('CpuStatusReportsValueListEntry', 7, repeated=True)
    deviceFiles = _messages.MessageField('DeviceFilesValueListEntry', 8, repeated=True)
    deviceId = _messages.StringField(9)
    diskVolumeReports = _messages.MessageField('DiskVolumeReportsValueListEntry', 10, repeated=True)
    dockMacAddress = _messages.StringField(11)
    etag = _messages.StringField(12)
    ethernetMacAddress = _messages.StringField(13)
    ethernetMacAddress0 = _messages.StringField(14)
    firmwareVersion = _messages.StringField(15)
    kind = _messages.StringField(16, default=u'admin#directory#chromeosdevice')
    lastEnrollmentTime = _message_types.DateTimeField(17)
    lastSync = _message_types.DateTimeField(18)
    macAddress = _messages.StringField(19)
    manufactureDate = _messages.StringField(20)
    meid = _messages.StringField(21)
    model = _messages.StringField(22)
    notes = _messages.StringField(23)
    orderNumber = _messages.StringField(24)
    orgUnitPath = _messages.StringField(25)
    osVersion = _messages.StringField(26)
    platformVersion = _messages.StringField(27)
    recentUsers = _messages.MessageField('RecentUsersValueListEntry', 28, repeated=True)
    serialNumber = _messages.StringField(29)
    status = _messages.StringField(30)
    supportEndDate = _message_types.DateTimeField(31)
    systemRamFreeReports = _messages.MessageField('SystemRamFreeReportsValueListEntry', 32, repeated=True)
    systemRamTotal = _messages.IntegerField(33)
    tpmVersionInfo = _messages.MessageField('TpmVersionInfoValue', 34)
    willAutoRenew = _messages.BooleanField(35)