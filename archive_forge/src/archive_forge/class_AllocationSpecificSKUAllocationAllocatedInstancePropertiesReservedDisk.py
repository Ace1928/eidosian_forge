from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationSpecificSKUAllocationAllocatedInstancePropertiesReservedDisk(_messages.Message):
    """A AllocationSpecificSKUAllocationAllocatedInstancePropertiesReservedDisk
  object.

  Enums:
    InterfaceValueValuesEnum: Specifies the disk interface to use for
      attaching this disk, which is either SCSI or NVME. The default is SCSI.
      For performance characteristics of SCSI over NVMe, see Local SSD
      performance.

  Fields:
    diskSizeGb: Specifies the size of the disk in base-2 GB.
    interface: Specifies the disk interface to use for attaching this disk,
      which is either SCSI or NVME. The default is SCSI. For performance
      characteristics of SCSI over NVMe, see Local SSD performance.
  """

    class InterfaceValueValuesEnum(_messages.Enum):
        """Specifies the disk interface to use for attaching this disk, which is
    either SCSI or NVME. The default is SCSI. For performance characteristics
    of SCSI over NVMe, see Local SSD performance.

    Values:
      NVME: <no description>
      SCSI: <no description>
    """
        NVME = 0
        SCSI = 1
    diskSizeGb = _messages.IntegerField(1)
    interface = _messages.EnumField('InterfaceValueValuesEnum', 2)