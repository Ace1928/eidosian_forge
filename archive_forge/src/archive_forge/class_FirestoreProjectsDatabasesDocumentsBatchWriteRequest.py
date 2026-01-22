from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsBatchWriteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsBatchWriteRequest object.

  Fields:
    batchWriteRequest: A BatchWriteRequest resource to be passed as the
      request body.
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`.
  """
    batchWriteRequest = _messages.MessageField('BatchWriteRequest', 1)
    database = _messages.StringField(2, required=True)