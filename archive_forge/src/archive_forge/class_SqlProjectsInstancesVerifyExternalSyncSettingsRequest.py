from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlProjectsInstancesVerifyExternalSyncSettingsRequest(_messages.Message):
    """A SqlProjectsInstancesVerifyExternalSyncSettingsRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
    sqlInstancesVerifyExternalSyncSettingsRequest: A
      SqlInstancesVerifyExternalSyncSettingsRequest resource to be passed as
      the request body.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sqlInstancesVerifyExternalSyncSettingsRequest = _messages.MessageField('SqlInstancesVerifyExternalSyncSettingsRequest', 3)