from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesUpdateRequest(_messages.Message):
    """A SqlInstancesUpdateRequest object.

  Fields:
    databaseInstance: A DatabaseInstance resource to be passed as the request
      body.
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    databaseInstance = _messages.MessageField('DatabaseInstance', 1)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)