from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrefixCounter(_messages.Message):
    """PrefixCounter contains a collection of prefixes related counts.

  Fields:
    advertised: Number of prefixes advertised.
    denied: Number of prefixes denied.
    received: Number of prefixes received.
    sent: Number of prefixes sent.
    suppressed: Number of prefixes suppressed.
    withdrawn: Number of prefixes withdrawn.
  """
    advertised = _messages.IntegerField(1)
    denied = _messages.IntegerField(2)
    received = _messages.IntegerField(3)
    sent = _messages.IntegerField(4)
    suppressed = _messages.IntegerField(5)
    withdrawn = _messages.IntegerField(6)