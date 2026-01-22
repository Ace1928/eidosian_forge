from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExcludeReplicas(_messages.Message):
    """An ExcludeReplicas contains a repeated set of ReplicaSelection that
  should be excluded from serving requests.

  Fields:
    replicaSelections: The directed read replica selector.
  """
    replicaSelections = _messages.MessageField('ReplicaSelection', 1, repeated=True)