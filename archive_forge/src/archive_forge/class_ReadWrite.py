from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadWrite(_messages.Message):
    """Options for a transaction that can be used to read and write documents.
  Firestore does not allow 3rd party auth requests to create read-write.
  transactions.

  Fields:
    retryTransaction: An optional transaction to retry.
  """
    retryTransaction = _messages.BytesField(1)