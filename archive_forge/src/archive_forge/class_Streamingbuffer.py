from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Streamingbuffer(_messages.Message):
    """A Streamingbuffer object.

  Fields:
    estimatedBytes: [Output-only] A lower-bound estimate of the number of
      bytes currently in the streaming buffer.
    estimatedRows: [Output-only] A lower-bound estimate of the number of rows
      currently in the streaming buffer.
    oldestEntryTime: [Output-only] Contains the timestamp of the oldest entry
      in the streaming buffer, in milliseconds since the epoch, if the
      streaming buffer is available.
  """
    estimatedBytes = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    estimatedRows = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    oldestEntryTime = _messages.IntegerField(3, variant=_messages.Variant.UINT64)