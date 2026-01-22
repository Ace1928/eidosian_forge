from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesRestoreBackupRequest(_messages.Message):
    """A SqlInstancesRestoreBackupRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    instancesRestoreBackupRequest: A InstancesRestoreBackupRequest resource to
      be passed as the request body.
    project: Project ID of the project that contains the instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesRestoreBackupRequest = _messages.MessageField('InstancesRestoreBackupRequest', 2)
    project = _messages.StringField(3, required=True)