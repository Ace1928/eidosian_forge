from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesFailoverRequest(_messages.Message):
    """A SqlInstancesFailoverRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    instancesFailoverRequest: A InstancesFailoverRequest resource to be passed
      as the request body.
    project: ID of the project that contains the read replica.
  """
    instance = _messages.StringField(1, required=True)
    instancesFailoverRequest = _messages.MessageField('InstancesFailoverRequest', 2)
    project = _messages.StringField(3, required=True)