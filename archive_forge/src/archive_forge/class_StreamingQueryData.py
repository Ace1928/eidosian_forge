from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingQueryData(_messages.Message):
    """Streaming

  Fields:
    endTimestamp: A string attribute.
    exception: A string attribute.
    isActive: A boolean attribute.
    name: A string attribute.
    runId: A string attribute.
    startTimestamp: A string attribute.
    streamingQueryId: A string attribute.
  """
    endTimestamp = _messages.IntegerField(1)
    exception = _messages.StringField(2)
    isActive = _messages.BooleanField(3)
    name = _messages.StringField(4)
    runId = _messages.StringField(5)
    startTimestamp = _messages.IntegerField(6)
    streamingQueryId = _messages.StringField(7)