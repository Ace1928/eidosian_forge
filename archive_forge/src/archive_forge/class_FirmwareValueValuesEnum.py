from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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