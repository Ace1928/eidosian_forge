from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlDatabasesUpdateRequest(_messages.Message):
    """A SqlDatabasesUpdateRequest object.

  Fields:
    database: Name of the database to be updated in the instance.
    databaseResource: A Database resource to be passed as the request body.
    instance: Database instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    database = _messages.StringField(1, required=True)
    databaseResource = _messages.MessageField('Database', 2)
    instance = _messages.StringField(3, required=True)
    project = _messages.StringField(4, required=True)