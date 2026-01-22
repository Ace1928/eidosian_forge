from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetVMDetails(_messages.Message):
    """TargetVMDetails is a collection of details for creating a VM in a target
  Compute Engine project.

  Enums:
    BootOptionValueValuesEnum: Output only. The VM Boot Option, as set in the
      source VM.
    DiskTypeValueValuesEnum: The disk type to use in the VM.
    LicenseTypeValueValuesEnum: The license type to use in OS adaptation.

  Messages:
    LabelsValue: A map of labels to associate with the VM.
    MetadataValue: The metadata key/value pairs to assign to the VM.

  Fields:
    appliedLicense: Output only. The OS license returned from the adaptation
      module report.
    bootOption: Output only. The VM Boot Option, as set in the source VM.
    computeScheduling: Compute instance scheduling information (if empty
      default is used).
    diskType: The disk type to use in the VM.
    externalIp: The external IP to define in the VM.
    internalIp: The internal IP to define in the VM. The formats accepted are:
      `ephemeral` \\ ipv4 address \\ a named address resource full path.
    labels: A map of labels to associate with the VM.
    licenseType: The license type to use in OS adaptation.
    machineType: The machine type to create the VM with.
    machineTypeSeries: The machine type series to create the VM with.
    metadata: The metadata key/value pairs to assign to the VM.
    name: The name of the VM to create.
    network: The network to connect the VM to.
    networkInterfaces: List of NICs connected to this VM.
    networkTags: A list of network tags to associate with the VM.
    project: Output only. The project in which to create the VM.
    secureBoot: Defines whether the instance has Secure Boot enabled. This can
      be set to true only if the vm boot option is EFI.
    serviceAccount: The service account to associate the VM with.
    subnetwork: The subnetwork to connect the VM to.
    targetProject: The full path of the resource of type TargetProject which
      represents the Compute Engine project in which to create this VM.
    zone: The zone in which to create the VM.
  """

    class BootOptionValueValuesEnum(_messages.Enum):
        """Output only. The VM Boot Option, as set in the source VM.

    Values:
      BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      EFI: The boot option is EFI.
      BIOS: The boot option is BIOS.
    """
        BOOT_OPTION_UNSPECIFIED = 0
        EFI = 1
        BIOS = 2

    class DiskTypeValueValuesEnum(_messages.Enum):
        """The disk type to use in the VM.

    Values:
      DISK_TYPE_UNSPECIFIED: An unspecified disk type. Will be used as
        STANDARD.
      STANDARD: A Standard disk type.
      BALANCED: An alternative to SSD persistent disks that balance
        performance and cost.
      SSD: SSD hard disk type.
    """
        DISK_TYPE_UNSPECIFIED = 0
        STANDARD = 1
        BALANCED = 2
        SSD = 3

    class LicenseTypeValueValuesEnum(_messages.Enum):
        """The license type to use in OS adaptation.

    Values:
      DEFAULT: The license type is the default for the OS.
      PAYG: The license type is Pay As You Go license type.
      BYOL: The license type is Bring Your Own License type.
    """
        DEFAULT = 0
        PAYG = 1
        BYOL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """A map of labels to associate with the VM.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """The metadata key/value pairs to assign to the VM.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    appliedLicense = _messages.MessageField('AppliedLicense', 1)
    bootOption = _messages.EnumField('BootOptionValueValuesEnum', 2)
    computeScheduling = _messages.MessageField('ComputeScheduling', 3)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 4)
    externalIp = _messages.StringField(5)
    internalIp = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    licenseType = _messages.EnumField('LicenseTypeValueValuesEnum', 8)
    machineType = _messages.StringField(9)
    machineTypeSeries = _messages.StringField(10)
    metadata = _messages.MessageField('MetadataValue', 11)
    name = _messages.StringField(12)
    network = _messages.StringField(13)
    networkInterfaces = _messages.MessageField('NetworkInterface', 14, repeated=True)
    networkTags = _messages.StringField(15, repeated=True)
    project = _messages.StringField(16)
    secureBoot = _messages.BooleanField(17)
    serviceAccount = _messages.StringField(18)
    subnetwork = _messages.StringField(19)
    targetProject = _messages.StringField(20)
    zone = _messages.StringField(21)