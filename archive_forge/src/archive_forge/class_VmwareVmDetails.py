from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVmDetails(_messages.Message):
    """VmwareVmDetails describes a VM in vCenter.

  Enums:
    BootOptionValueValuesEnum: Output only. The VM Boot Option.
    PowerStateValueValuesEnum: The power state of the VM at the moment list
      was taken.

  Fields:
    bootOption: Output only. The VM Boot Option.
    committedStorageMb: The total size of the storage allocated to the VM in
      MB.
    cpuCount: The number of cpus in the VM.
    datacenterDescription: The descriptive name of the vCenter's datacenter
      this VM is contained in.
    datacenterId: The id of the vCenter's datacenter this VM is contained in.
    diskCount: The number of disks the VM has.
    displayName: The display name of the VM. Note that this is not necessarily
      unique.
    guestDescription: The VM's OS. See for example https://vdc-
      repo.vmware.com/vmwb-repository/dcr-public/da47f910-60ac-438b-8b9b-
      6122f4d14524/16b7274a-bf8b-4b4c-a05e-
      746f2aa93c8c/doc/vim.vm.GuestOsDescriptor.GuestOsIdentifier.html for
      types of strings this might hold.
    memoryMb: The size of the memory of the VM in MB.
    powerState: The power state of the VM at the moment list was taken.
    uuid: The unique identifier of the VM in vCenter.
    vmId: The VM's id in the source (note that this is not the MigratingVm's
      id). This is the moref id of the VM.
  """

    class BootOptionValueValuesEnum(_messages.Enum):
        """Output only. The VM Boot Option.

    Values:
      BOOT_OPTION_UNSPECIFIED: The boot option is unknown.
      EFI: The boot option is EFI.
      BIOS: The boot option is BIOS.
    """
        BOOT_OPTION_UNSPECIFIED = 0
        EFI = 1
        BIOS = 2

    class PowerStateValueValuesEnum(_messages.Enum):
        """The power state of the VM at the moment list was taken.

    Values:
      POWER_STATE_UNSPECIFIED: Power state is not specified.
      ON: The VM is turned ON.
      OFF: The VM is turned OFF.
      SUSPENDED: The VM is suspended. This is similar to hibernation or sleep
        mode.
    """
        POWER_STATE_UNSPECIFIED = 0
        ON = 1
        OFF = 2
        SUSPENDED = 3
    bootOption = _messages.EnumField('BootOptionValueValuesEnum', 1)
    committedStorageMb = _messages.IntegerField(2)
    cpuCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    datacenterDescription = _messages.StringField(4)
    datacenterId = _messages.StringField(5)
    diskCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    displayName = _messages.StringField(7)
    guestDescription = _messages.StringField(8)
    memoryMb = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    powerState = _messages.EnumField('PowerStateValueValuesEnum', 10)
    uuid = _messages.StringField(11)
    vmId = _messages.StringField(12)