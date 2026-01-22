from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionStyleValueValuesEnum(_messages.Enum):
    """Optional. The structure of paths containing partition data within the
    entity.

    Values:
      PARTITION_STYLE_UNSPECIFIED: PartitionStyle unspecified
      HIVE_COMPATIBLE: Partitions are hive-compatible. Examples:
        gs://bucket/path/to/table/dt=2019-10-31/lang=en,
        gs://bucket/path/to/table/dt=2019-10-31/lang=en/late.
    """
    PARTITION_STYLE_UNSPECIFIED = 0
    HIVE_COMPATIBLE = 1