from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecoveryConfig(_messages.Message):
    """The Recovery settings of an environment.

  Fields:
    scheduledSnapshotsConfig: Optional. The configuration for scheduled
      snapshot creation mechanism.
  """
    scheduledSnapshotsConfig = _messages.MessageField('ScheduledSnapshotsConfig', 1)