from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataDisk(_messages.Message):
    """An instance-attached disk resource.

  Enums:
    DiskEncryptionValueValuesEnum: Optional. Input only. Disk encryption
      method used on the boot and data disks, defaults to GMEK.
    DiskTypeValueValuesEnum: Optional. Input only. Indicates the type of the
      disk.

  Fields:
    diskEncryption: Optional. Input only. Disk encryption method used on the
      boot and data disks, defaults to GMEK.
    diskSizeGb: Optional. The size of the disk in GB attached to this VM
      instance, up to a maximum of 64000 GB (64 TB). If not specified, this
      defaults to 100.
    diskType: Optional. Input only. Indicates the type of the disk.
    kmsKey: Optional. Input only. The KMS key used to encrypt the disks, only
      applicable if disk_encryption is CMEK. Format: `projects/{project_id}/lo
      cations/{location}/keyRings/{key_ring_id}/cryptoKeys/{key_id}` Learn
      more about using your own encryption keys.
  """

    class DiskEncryptionValueValuesEnum(_messages.Enum):
        """Optional. Input only. Disk encryption method used on the boot and data
    disks, defaults to GMEK.

    Values:
      DISK_ENCRYPTION_UNSPECIFIED: Disk encryption is not specified.
      GMEK: Use Google managed encryption keys to encrypt the boot disk.
      CMEK: Use customer managed encryption keys to encrypt the boot disk.
    """
        DISK_ENCRYPTION_UNSPECIFIED = 0
        GMEK = 1
        CMEK = 2

    class DiskTypeValueValuesEnum(_messages.Enum):
        """Optional. Input only. Indicates the type of the disk.

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
    diskEncryption = _messages.EnumField('DiskEncryptionValueValuesEnum', 1)
    diskSizeGb = _messages.IntegerField(2)
    diskType = _messages.EnumField('DiskTypeValueValuesEnum', 3)
    kmsKey = _messages.StringField(4)