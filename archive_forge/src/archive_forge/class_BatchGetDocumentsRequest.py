from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetDocumentsRequest(_messages.Message):
    """The request for Firestore.BatchGetDocuments.

  Fields:
    documents: The names of the documents to retrieve. In the format:
      `projects/{project_id}/databases/{database_id}/documents/{document_path}
      `. The request will fail if any of the document is not a child resource
      of the given `database`. Duplicate names will be elided.
    mask: The fields to return. If not set, returns all fields. If a document
      has a field that is not present in this mask, that field will not be
      returned in the response.
    newTransaction: Starts a new transaction and reads the documents. Defaults
      to a read-only transaction. The new transaction ID will be returned as
      the first response in the stream.
    readTime: Reads documents as they were at the given time. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    transaction: Reads documents in a transaction.
  """
    documents = _messages.StringField(1, repeated=True)
    mask = _messages.MessageField('DocumentMask', 2)
    newTransaction = _messages.MessageField('TransactionOptions', 3)
    readTime = _messages.StringField(4)
    transaction = _messages.BytesField(5)