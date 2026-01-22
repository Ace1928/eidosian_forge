from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteResponse(_messages.Message):
    """The response for Firestore.Write.

  Fields:
    commitTime: The time at which the commit occurred. Any read with an equal
      or greater `read_time` is guaranteed to see the effects of the write.
    streamId: The ID of the stream. Only set on the first message, when a new
      stream was created.
    streamToken: A token that represents the position of this response in the
      stream. This can be used by a client to resume the stream at this point.
      This field is always set.
    writeResults: The result of applying the writes. This i-th write result
      corresponds to the i-th write in the request.
  """
    commitTime = _messages.StringField(1)
    streamId = _messages.StringField(2)
    streamToken = _messages.BytesField(3)
    writeResults = _messages.MessageField('WriteResult', 4, repeated=True)