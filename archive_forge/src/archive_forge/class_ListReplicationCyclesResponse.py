from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReplicationCyclesResponse(_messages.Message):
    """Response message for 'ListReplicationCycles' request.

  Fields:
    nextPageToken: Output only. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    replicationCycles: Output only. The list of replication cycles response.
    unreachable: Output only. Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    replicationCycles = _messages.MessageField('ReplicationCycle', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)