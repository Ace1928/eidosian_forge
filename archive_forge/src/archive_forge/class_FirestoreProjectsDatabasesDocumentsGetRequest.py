from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirestoreProjectsDatabasesDocumentsGetRequest(_messages.Message):
    """A FirestoreProjectsDatabasesDocumentsGetRequest object.

  Fields:
    mask_fieldPaths: The list of field paths in the mask. See Document.fields
      for a field path syntax reference.
    name: Required. The resource name of the Document to get. In the format:
      `projects/{project_id}/databases/{database_id}/documents/{document_path}
      `.
    readTime: Reads the version of the document at the given time. This must
      be a microsecond precision timestamp within the past one hour, or if
      Point-in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    transaction: Reads the document in a transaction.
  """
    mask_fieldPaths = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2, required=True)
    readTime = _messages.StringField(3)
    transaction = _messages.BytesField(4)