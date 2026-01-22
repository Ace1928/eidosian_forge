from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesSessionsCommitRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesSessionsCommitRequest object.

  Fields:
    commitRequest: A CommitRequest resource to be passed as the request body.
    session: Required. The session in which the transaction to be committed is
      running.
  """
    commitRequest = _messages.MessageField('CommitRequest', 1)
    session = _messages.StringField(2, required=True)