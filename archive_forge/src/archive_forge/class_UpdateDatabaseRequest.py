from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateDatabaseRequest(_messages.Message):
    """The request for UpdateDatabase.

  Fields:
    database: Required. The database to update. The `name` field of the
      database is of the form `projects//instances//databases/`.
    updateMask: Required. The list of fields to update. Currently, only
      `enable_drop_protection` field can be updated.
  """
    database = _messages.MessageField('Database', 1)
    updateMask = _messages.StringField(2)