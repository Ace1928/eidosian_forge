from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaReplayResultsSummary(_messages.Message):
    """Summary statistics about the replayed log entries.

  Fields:
    differenceCount: The number of replayed log entries with a difference
      between baseline and simulated policies.
    errorCount: The number of log entries that could not be replayed.
    logCount: The total number of log entries replayed.
    newestDate: The date of the newest log entry replayed.
    oldestDate: The date of the oldest log entry replayed.
    unchangedCount: The number of replayed log entries with no difference
      between baseline and simulated policies.
  """
    differenceCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    errorCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    logCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    newestDate = _messages.MessageField('GoogleTypeDate', 4)
    oldestDate = _messages.MessageField('GoogleTypeDate', 5)
    unchangedCount = _messages.IntegerField(6, variant=_messages.Variant.INT32)