from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlDatabasesDeleteRequest(_messages.Message):
    """A SqlDatabasesDeleteRequest object.

  Fields:
    database: Name of the database to be deleted in the instance.
    instance: Database instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    database = _messages.StringField(1, required=True)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)