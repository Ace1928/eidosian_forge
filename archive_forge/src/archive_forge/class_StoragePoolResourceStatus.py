from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragePoolResourceStatus(_messages.Message):
    """[Output Only] Contains output only fields.

  Fields:
    diskCount: [Output Only] Number of disks used.
    lastResizeTimestamp: [Output Only] Timestamp of the last successful resize
      in RFC3339 text format.
    maxTotalProvisionedDiskCapacityGb: [Output Only] Maximum allowed aggregate
      disk size in gigabytes.
    poolUsedCapacityBytes: [Output Only] Space used by data stored in disks
      within the storage pool (in bytes). This will reflect the total number
      of bytes written to the disks in the pool, in contrast to the capacity
      of those disks.
    poolUsedIops: Sum of all the disks' provisioned IOPS, minus some amount
      that is allowed per disk that is not counted towards pool's IOPS
      capacity.
    poolUsedThroughput: [Output Only] Sum of all the disks' provisioned
      throughput in MB/s.
    poolUserWrittenBytes: [Output Only] Amount of data written into the pool,
      before it is compacted.
    totalProvisionedDiskCapacityGb: [Output Only] Sum of all the capacity
      provisioned in disks in this storage pool. A disk's provisioned capacity
      is the same as its total capacity.
    totalProvisionedDiskIops: [Output Only] Sum of all the disks' provisioned
      IOPS.
    totalProvisionedDiskThroughput: [Output Only] Sum of all the disks'
      provisioned throughput in MB/s, minus some amount that is allowed per
      disk that is not counted towards pool's throughput capacity.
  """
    diskCount = _messages.IntegerField(1)
    lastResizeTimestamp = _messages.StringField(2)
    maxTotalProvisionedDiskCapacityGb = _messages.IntegerField(3)
    poolUsedCapacityBytes = _messages.IntegerField(4)
    poolUsedIops = _messages.IntegerField(5)
    poolUsedThroughput = _messages.IntegerField(6)
    poolUserWrittenBytes = _messages.IntegerField(7)
    totalProvisionedDiskCapacityGb = _messages.IntegerField(8)
    totalProvisionedDiskIops = _messages.IntegerField(9)
    totalProvisionedDiskThroughput = _messages.IntegerField(10)