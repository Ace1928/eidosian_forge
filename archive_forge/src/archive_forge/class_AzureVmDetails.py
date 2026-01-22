from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureVmDetails(_messages.Message):
    """AzureVmDetails describes a VM in Azure.

  Enums:
    BootOptionValueValuesEnum: The VM Boot Option.
    PowerStateValueValuesEnum: The power state of the VM at the moment list
      was taken.

  Messages:
    TagsValue: The tags of the VM.

  Fields:
    bootOption: The VM Boot Option.
    committedStorageMb: The total size of the storage allocated to the VM in
      MB.
    computerName: The VM's ComputerName.
    cpuCount: The number of cpus the VM has.
    diskCount: The number of disks the VM has, including OS disk.
    disks: Description of the data disks.
    memoryMb: The memory size of the VM in MB.
    osDescription: Description of the OS.
    osDisk: Description of the OS disk.
    powerState: The power state of the VM at the moment list was taken.
    tags: The tags of the VM.
    vmId: The VM full path in Azure.
    vmSize: VM size as configured in Azure. Determines the VM's hardware spec.
  """

    class BootOptionValueValuesEnum(_messages.Enum):
        """The VM Boot Option.

    Values:
      BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      EFI: The boot option is UEFI.
      BIOS: The boot option is BIOS.
    """
        BOOT_OPTION_UNSPECIFIED = 0
        EFI = 1
        BIOS = 2

    class PowerStateValueValuesEnum(_messages.Enum):
        """The power state of the VM at the moment list was taken.

    Values:
      POWER_STATE_UNSPECIFIED: Power state is not specified.
      STARTING: The VM is starting.
      RUNNING: The VM is running.
      STOPPING: The VM is stopping.
      STOPPED: The VM is stopped.
      DEALLOCATING: The VM is deallocating.
      DEALLOCATED: The VM is deallocated.
      UNKNOWN: The VM's power state is unknown.
    """
        POWER_STATE_UNSPECIFIED = 0
        STARTING = 1
        RUNNING = 2
        STOPPING = 3
        STOPPED = 4
        DEALLOCATING = 5
        DEALLOCATED = 6
        UNKNOWN = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """The tags of the VM.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bootOption = _messages.EnumField('BootOptionValueValuesEnum', 1)
    committedStorageMb = _messages.IntegerField(2)
    computerName = _messages.StringField(3)
    cpuCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    diskCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    disks = _messages.MessageField('Disk', 6, repeated=True)
    memoryMb = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    osDescription = _messages.MessageField('OSDescription', 8)
    osDisk = _messages.MessageField('OSDisk', 9)
    powerState = _messages.EnumField('PowerStateValueValuesEnum', 10)
    tags = _messages.MessageField('TagsValue', 11)
    vmId = _messages.StringField(12)
    vmSize = _messages.StringField(13)