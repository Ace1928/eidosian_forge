from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesStopReplicaRequest(_messages.Message):
    """A SqlInstancesStopReplicaRequest object.

  Fields:
    instance: Cloud SQL read replica instance name.
    project: ID of the project that contains the read replica.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)