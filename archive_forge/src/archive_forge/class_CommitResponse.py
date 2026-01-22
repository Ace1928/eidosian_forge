from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitResponse(_messages.Message):
    """The response for Firestore.Commit.

  Fields:
    commitTime: The time at which the commit occurred. Any read with an equal
      or greater `read_time` is guaranteed to see the effects of the commit.
    writeResults: The result of applying the writes. This i-th write result
      corresponds to the i-th write in the request.
  """
    commitTime = _messages.StringField(1)
    writeResults = _messages.MessageField('WriteResult', 2, repeated=True)