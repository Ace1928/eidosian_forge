from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogSplit(_messages.Message):
    """Additional information used to correlate multiple log entries. Used when
  a single LogEntry would exceed the Google Cloud Logging size limit and is
  split across multiple log entries.

  Fields:
    index: The index of this LogEntry in the sequence of split log entries.
      Log entries are given |index| values 0, 1, ..., n-1 for a sequence of n
      log entries.
    totalSplits: The total number of log entries that the original LogEntry
      was split into.
    uid: A globally unique identifier for all log entries in a sequence of
      split log entries. All log entries with the same |LogSplit.uid| are
      assumed to be part of the same sequence of split log entries.
  """
    index = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    totalSplits = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    uid = _messages.StringField(3)