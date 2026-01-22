from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SnapshotReservationDetail(_messages.Message):
    """Details about snapshot space reservation and usage on the storage
  volume.

  Fields:
    reservedSpaceGib: The space on this storage volume reserved for snapshots,
      shown in GiB.
    reservedSpacePercent: Percent of the total Volume size reserved for
      snapshot copies. Enabling snapshots requires reserving 20% or more of
      the storage volume space for snapshots. Maximum reserved space for
      snapshots is 40%. Setting this field will effectively set
      snapshot_enabled to true.
    reservedSpaceRemainingGib: The amount, in GiB, of available space in this
      storage volume's reserved snapshot space.
    reservedSpaceUsedPercent: The percent of snapshot space on this storage
      volume actually being used by the snapshot copies. This value might be
      higher than 100% if the snapshot copies have overflowed into the data
      portion of the storage volume.
  """
    reservedSpaceGib = _messages.IntegerField(1)
    reservedSpacePercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    reservedSpaceRemainingGib = _messages.IntegerField(3)
    reservedSpaceUsedPercent = _messages.IntegerField(4, variant=_messages.Variant.INT32)