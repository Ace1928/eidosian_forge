from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnUpdateActionValueValuesEnum(_messages.Enum):
    """Specifies which action to take on instance update with this disk.
    Default is to use the existing disk.

    Values:
      RECREATE_DISK: Always recreate the disk.
      RECREATE_DISK_IF_SOURCE_CHANGED: Recreate the disk if source (image,
        snapshot) of this disk is different from source of existing disk.
      USE_EXISTING_DISK: Use the existing disk, this is the default behaviour.
    """
    RECREATE_DISK = 0
    RECREATE_DISK_IF_SOURCE_CHANGED = 1
    USE_EXISTING_DISK = 2