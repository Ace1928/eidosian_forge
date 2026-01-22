from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesInsertRequest(_messages.Message):
    """A SqlInstancesInsertRequest object.

  Fields:
    databaseInstance: A DatabaseInstance resource to be passed as the request
      body.
    project: Project ID of the project to which the newly created Cloud SQL
      instances should belong.
  """
    databaseInstance = _messages.MessageField('DatabaseInstance', 1)
    project = _messages.StringField(2, required=True)