from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GqlQueryParameter(_messages.Message):
    """A binding parameter for a GQL query.

  Fields:
    cursor: A query cursor. Query cursors are returned in query result
      batches.
    value: A value parameter.
  """
    cursor = _messages.BytesField(1)
    value = _messages.MessageField('Value', 2)