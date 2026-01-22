from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDocumentsResponse(_messages.Message):
    """The response for Firestore.ListDocuments.

  Fields:
    documents: The Documents found.
    nextPageToken: A token to retrieve the next page of documents. If this
      field is omitted, there are no subsequent pages.
  """
    documents = _messages.MessageField('Document', 1, repeated=True)
    nextPageToken = _messages.StringField(2)