from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Straggler(_messages.Message):
    """Information for a straggler.

  Fields:
    batchStraggler: Batch straggler identification and debugging information.
    streamingStraggler: Streaming straggler identification and debugging
      information.
  """
    batchStraggler = _messages.MessageField('StragglerInfo', 1)
    streamingStraggler = _messages.MessageField('StreamingStragglerInfo', 2)