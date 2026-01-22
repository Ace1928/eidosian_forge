from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GceRegionalPersistentDisk(_messages.Message):
    """A Persistent Directory backed by a Compute Engine regional persistent
  disk. The persistent_directories field is repeated, but it may contain only
  one entry. It creates a [persistent
  disk](https://cloud.google.com/compute/docs/disks/persistent-disks) that
  mounts to the workstation VM at `/home` when the session starts and detaches
  when the session ends. If this field is empty, workstations created with
  this configuration do not have a persistent home directory.

  Enums:
    ReclaimPolicyValueValuesEnum: Optional. Whether the persistent disk should
      be deleted when the workstation is deleted. Valid values are `DELETE`
      and `RETAIN`. Defaults to `DELETE`.

  Fields:
    diskType: Optional. The [type of the persistent
      disk](https://cloud.google.com/compute/docs/disks#disk-types) for the
      home directory. Defaults to `"pd-standard"`.
    fsType: Optional. Type of file system that the disk should be formatted
      with. The workstation image must support this file system type. Must be
      empty if source_snapshot is set. Defaults to `"ext4"`.
    reclaimPolicy: Optional. Whether the persistent disk should be deleted
      when the workstation is deleted. Valid values are `DELETE` and `RETAIN`.
      Defaults to `DELETE`.
    sizeGb: Optional. The GB capacity of a persistent home directory for each
      workstation created with this configuration. Must be empty if
      source_snapshot is set. Valid values are `10`, `50`, `100`, `200`,
      `500`, or `1000`. Defaults to `200`. If less than `200` GB, the
      disk_type must be `"pd-balanced"` or `"pd-ssd"`.
    sourceSnapshot: Optional. Name of the snapshot to use as the source for
      the disk. If set, size_gb and fs_type must be empty.
  """

    class ReclaimPolicyValueValuesEnum(_messages.Enum):
        """Optional. Whether the persistent disk should be deleted when the
    workstation is deleted. Valid values are `DELETE` and `RETAIN`. Defaults
    to `DELETE`.

    Values:
      RECLAIM_POLICY_UNSPECIFIED: Do not use.
      DELETE: Delete the persistent disk when deleting the workstation.
      RETAIN: Keep the persistent disk when deleting the workstation. An
        administrator must manually delete the disk.
    """
        RECLAIM_POLICY_UNSPECIFIED = 0
        DELETE = 1
        RETAIN = 2
    diskType = _messages.StringField(1)
    fsType = _messages.StringField(2)
    reclaimPolicy = _messages.EnumField('ReclaimPolicyValueValuesEnum', 3)
    sizeGb = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    sourceSnapshot = _messages.StringField(5)