from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskAsyncReplication(_messages.Message):
    """A DiskAsyncReplication object.

  Fields:
    consistencyGroupPolicy: [Output Only] URL of the
      DiskConsistencyGroupPolicy if replication was started on the disk as a
      member of a group.
    consistencyGroupPolicyId: [Output Only] ID of the
      DiskConsistencyGroupPolicy if replication was started on the disk as a
      member of a group.
    disk: The other disk asynchronously replicated to or from the current
      disk. You can provide this as a partial or full URL to the resource. For
      example, the following are valid values: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /disks/disk - projects/project/zones/zone/disks/disk -
      zones/zone/disks/disk
    diskId: [Output Only] The unique ID of the other disk asynchronously
      replicated to or from the current disk. This value identifies the exact
      disk that was used to create this replication. For example, if you
      started replicating the persistent disk from a disk that was later
      deleted and recreated under the same name, the disk ID would identify
      the exact version of the disk that was used.
  """
    consistencyGroupPolicy = _messages.StringField(1)
    consistencyGroupPolicyId = _messages.StringField(2)
    disk = _messages.StringField(3)
    diskId = _messages.StringField(4)