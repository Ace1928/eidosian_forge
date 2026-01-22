from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesRotateServerCaRequest(_messages.Message):
    """A SqlInstancesRotateServerCaRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    instancesRotateServerCaRequest: A InstancesRotateServerCaRequest resource
      to be passed as the request body.
    project: Project ID of the project that contains the instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesRotateServerCaRequest = _messages.MessageField('InstancesRotateServerCaRequest', 2)
    project = _messages.StringField(3, required=True)