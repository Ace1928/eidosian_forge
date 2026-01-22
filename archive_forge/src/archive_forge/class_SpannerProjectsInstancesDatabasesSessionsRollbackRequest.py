from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsRollbackRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsRollbackRequest object.

  Fields:
    rollbackRequest: A RollbackRequest resource to be passed as the request
      body.
    session: Required. The session in which the transaction to roll back is
      running.
  """
    rollbackRequest = _messages.MessageField('RollbackRequest', 1)
    session = _messages.StringField(2, required=True)