from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsWriteRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsWriteRequest object.

  Fields:
    database: Required. The database name. In the format:
      `projects/{project_id}/databases/{database_id}`. This is only required
      in the first message.
    writeRequest: A WriteRequest resource to be passed as the request body.
  """
    database = _messages.StringField(1, required=True)
    writeRequest = _messages.MessageField('WriteRequest', 2)