from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BootDiskTypeValueValuesEnum(_messages.Enum):
    """Input only. The type of the boot disk attached to this instance,
    defaults to standard persistent disk (`PD_STANDARD`).

    Values:
      DISK_TYPE_UNSPECIFIED: Disk type not set.
      PD_STANDARD: Standard persistent disk type.
      PD_SSD: SSD persistent disk type.
      PD_BALANCED: Balanced persistent disk type.
      PD_EXTREME: Extreme persistent disk type.
    """
    DISK_TYPE_UNSPECIFIED = 0
    PD_STANDARD = 1
    PD_SSD = 2
    PD_BALANCED = 3
    PD_EXTREME = 4