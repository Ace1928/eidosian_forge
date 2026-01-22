from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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