from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackRequest(_messages.Message):
    """The request for Firestore.Rollback.

  Fields:
    transaction: Required. The transaction to roll back.
  """
    transaction = _messages.BytesField(1)