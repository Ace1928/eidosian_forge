from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskAsyncReplicationList(_messages.Message):
    """A DiskAsyncReplicationList object.

  Fields:
    asyncReplicationDisk: A DiskAsyncReplication attribute.
  """
    asyncReplicationDisk = _messages.MessageField('DiskAsyncReplication', 1)