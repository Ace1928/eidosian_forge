from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunQueryRequest(_messages.Message):
    """The request for Firestore.RunQuery.

  Fields:
    explainOptions: Optional. Explain options for the query. If set,
      additional query statistics will be returned. If not, only query results
      will be returned.
    newTransaction: Starts a new transaction and reads the documents. Defaults
      to a read-only transaction. The new transaction ID will be returned as
      the first response in the stream.
    readTime: Reads documents as they were at the given time. This must be a
      microsecond precision timestamp within the past one hour, or if Point-
      in-Time Recovery is enabled, can additionally be a whole minute
      timestamp within the past 7 days.
    structuredQuery: A structured query.
    transaction: Run the query within an already active transaction. The value
      here is the opaque transaction ID to execute the query in.
  """
    explainOptions = _messages.MessageField('ExplainOptions', 1)
    newTransaction = _messages.MessageField('TransactionOptions', 2)
    readTime = _messages.StringField(3)
    structuredQuery = _messages.MessageField('StructuredQuery', 4)
    transaction = _messages.BytesField(5)