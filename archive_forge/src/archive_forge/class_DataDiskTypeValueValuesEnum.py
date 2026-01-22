from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataDiskTypeValueValuesEnum(_messages.Enum):
    """The type of storage: `PD_SSD` (default) or `PD_HDD`.

    Values:
      SQL_DATA_DISK_TYPE_UNSPECIFIED: Unspecified.
      PD_SSD: SSD disk.
      PD_HDD: HDD disk.
    """
    SQL_DATA_DISK_TYPE_UNSPECIFIED = 0
    PD_SSD = 1
    PD_HDD = 2