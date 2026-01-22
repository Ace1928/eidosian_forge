from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkUtilization(_messages.Message):
    """Target scaling by network usage. Only applicable in the App Engine
  flexible environment.

  Fields:
    targetReceivedBytesPerSecond: Target bytes received per second.
    targetReceivedPacketsPerSecond: Target packets received per second.
    targetSentBytesPerSecond: Target bytes sent per second.
    targetSentPacketsPerSecond: Target packets sent per second.
  """
    targetReceivedBytesPerSecond = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    targetReceivedPacketsPerSecond = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    targetSentBytesPerSecond = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    targetSentPacketsPerSecond = _messages.IntegerField(4, variant=_messages.Variant.INT32)