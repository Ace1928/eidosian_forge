from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesTruncateLogRequest(_messages.Message):
    """A SqlInstancesTruncateLogRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    instancesTruncateLogRequest: A InstancesTruncateLogRequest resource to be
      passed as the request body.
    project: Project ID of the Cloud SQL project.
  """
    instance = _messages.StringField(1, required=True)
    instancesTruncateLogRequest = _messages.MessageField('InstancesTruncateLogRequest', 2)
    project = _messages.StringField(3, required=True)