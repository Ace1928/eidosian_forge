from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsSourceVmDetails(_messages.Message):
    """Represent the source AWS VM details.

  Enums:
    FirmwareValueValuesEnum: The firmware type of the source VM.

  Fields:
    committedStorageBytes: The total size of the disks being migrated in
      bytes.
    disks: The disks attached to the source VM.
    firmware: The firmware type of the source VM.
    vmCapabilitiesInfo: Output only. Information about VM capabilities needed
      for some Compute Engine features.
  """

    class FirmwareValueValuesEnum(_messages.Enum):
        """The firmware type of the source VM.

    Values:
      FIRMWARE_UNSPECIFIED: The firmware is unknown.
      EFI: The firmware is EFI.
      BIOS: The firmware is BIOS.
    """
        FIRMWARE_UNSPECIFIED = 0
        EFI = 1
        BIOS = 2
    committedStorageBytes = _messages.IntegerField(1)
    disks = _messages.MessageField('AwsDiskDetails', 2, repeated=True)
    firmware = _messages.EnumField('FirmwareValueValuesEnum', 3)
    vmCapabilitiesInfo = _messages.MessageField('VmCapabilities', 4)