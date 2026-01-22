from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopReplicationRequest(_messages.Message):
    """StopReplicationRequest stops a replication until resumed.

  Fields:
    force: Indicates whether to stop replication forcefully while data
      transfer is in progress. Warning! if force is true, this will abort any
      current transfers and can lead to data loss due to partial transfer. If
      force is false, stop replication will fail while data transfer is in
      progress and you will need to retry later.
  """
    force = _messages.BooleanField(1)