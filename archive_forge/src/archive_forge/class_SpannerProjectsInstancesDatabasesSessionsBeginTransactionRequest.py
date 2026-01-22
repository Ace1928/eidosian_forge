from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsBeginTransactionRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsBeginTransactionRequest
  object.

  Fields:
    beginTransactionRequest: A BeginTransactionRequest resource to be passed
      as the request body.
    session: Required. The session in which the transaction runs.
  """
    beginTransactionRequest = _messages.MessageField('BeginTransactionRequest', 1)
    session = _messages.StringField(2, required=True)