from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleClusterRouting(_messages.Message):
    """Unconditionally routes all read/write requests to a specific cluster.
  This option preserves read-your-writes consistency but does not improve
  availability.

  Fields:
    allowTransactionalWrites: Whether or not `CheckAndMutateRow` and
      `ReadModifyWriteRow` requests are allowed by this app profile. It is
      unsafe to send these requests to the same table/row/column in multiple
      clusters.
    clusterId: The cluster to which read/write requests should be routed.
  """
    allowTransactionalWrites = _messages.BooleanField(1)
    clusterId = _messages.StringField(2)