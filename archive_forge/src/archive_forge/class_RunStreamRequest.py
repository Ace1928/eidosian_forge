from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunStreamRequest(_messages.Message):
    """Request message for running a stream.

  Fields:
    cdcStrategy: Optional. The CDC strategy of the stream. If not set, the
      system's default value will be used.
  """
    cdcStrategy = _messages.MessageField('CdcStrategy', 1)