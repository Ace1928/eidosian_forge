from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListDocumentsResponse(_messages.Message):
    """Response message for Documents.ListDocuments.

  Fields:
    documents: The list of documents.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    documents = _messages.MessageField('GoogleCloudDialogflowV2Document', 1, repeated=True)
    nextPageToken = _messages.StringField(2)