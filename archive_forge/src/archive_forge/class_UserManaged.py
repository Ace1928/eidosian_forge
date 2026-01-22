from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserManaged(_messages.Message):
    """A replication policy that replicates the Secret payload into the
  locations specified in Secret.replication.user_managed.replicas

  Fields:
    replicas: Required. The list of Replicas for this Secret. Cannot be empty.
  """
    replicas = _messages.MessageField('Replica', 1, repeated=True)