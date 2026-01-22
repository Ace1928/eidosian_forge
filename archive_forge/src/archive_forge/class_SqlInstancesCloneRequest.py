from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesCloneRequest(_messages.Message):
    """A SqlInstancesCloneRequest object.

  Fields:
    instance: The ID of the Cloud SQL instance to be cloned (source). This
      does not include the project ID.
    instancesCloneRequest: A InstancesCloneRequest resource to be passed as
      the request body.
    project: Project ID of the source as well as the clone Cloud SQL instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesCloneRequest = _messages.MessageField('InstancesCloneRequest', 2)
    project = _messages.StringField(3, required=True)