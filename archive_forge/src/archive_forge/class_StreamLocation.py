from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamLocation(_messages.Message):
    """Describes a stream of data, either as input to be processed or as output
  of a streaming Dataflow job.

  Fields:
    customSourceLocation: The stream is a custom source.
    pubsubLocation: The stream is a pubsub stream.
    sideInputLocation: The stream is a streaming side input.
    streamingStageLocation: The stream is part of another computation within
      the current streaming Dataflow job.
  """
    customSourceLocation = _messages.MessageField('CustomSourceLocation', 1)
    pubsubLocation = _messages.MessageField('PubsubLocation', 2)
    sideInputLocation = _messages.MessageField('StreamingSideInputLocation', 3)
    streamingStageLocation = _messages.MessageField('StreamingStageLocation', 4)