from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeginTransactionResponse(_messages.Message):
    """The response for Firestore.BeginTransaction.

  Fields:
    transaction: The transaction that was started.
  """
    transaction = _messages.BytesField(1)