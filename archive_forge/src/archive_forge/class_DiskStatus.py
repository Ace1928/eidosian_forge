from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskStatus(_messages.Message):
    """The status of a disk on a VM.

  Fields:
    freeSpaceBytes: Free disk space.
    totalSpaceBytes: Total disk space.
  """
    freeSpaceBytes = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    totalSpaceBytes = _messages.IntegerField(2, variant=_messages.Variant.UINT64)