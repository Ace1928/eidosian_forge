from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsBeginTransactionRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsBeginTransactionRequest object.

  Fields:
    beginTransactionRequest: A BeginTransactionRequest resource to be passed
      as the request body.
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`.
  """
    beginTransactionRequest = _messages.MessageField('BeginTransactionRequest', 1)
    database = _messages.StringField(2, required=True)