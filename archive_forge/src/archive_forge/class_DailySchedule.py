from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DailySchedule(_messages.Message):
    """Make a snapshot every day e.g. at 04:00, 05:20, 23:50

  Fields:
    hour: Set the hour to start the snapshot (0-23), defaults to midnight (0).
    minute: Set the minute of the hour to start the snapshot (0-59), defaults
      to the top of the hour (0).
    snapshotsToKeep: The maximum number of Snapshots to keep for the hourly
      schedule
  """
    hour = _messages.FloatField(1)
    minute = _messages.FloatField(2)
    snapshotsToKeep = _messages.FloatField(3)