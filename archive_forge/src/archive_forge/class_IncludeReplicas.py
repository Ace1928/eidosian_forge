from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IncludeReplicas(_messages.Message):
    """An IncludeReplicas contains a repeated set of ReplicaSelection which
  indicates the order in which replicas should be considered.

  Fields:
    autoFailoverDisabled: If true, Spanner will not route requests to a
      replica outside the include_replicas list when all of the specified
      replicas are unavailable or unhealthy. Default value is `false`.
    replicaSelections: The directed read replica selector.
  """
    autoFailoverDisabled = _messages.BooleanField(1)
    replicaSelections = _messages.MessageField('ReplicaSelection', 2, repeated=True)