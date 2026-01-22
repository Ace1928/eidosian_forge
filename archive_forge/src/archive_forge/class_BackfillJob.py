from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackfillJob(_messages.Message):
    """Represents a backfill job on a specific stream object.

  Enums:
    StateValueValuesEnum: Output only. Backfill job state.
    TriggerValueValuesEnum: Backfill job's triggering reason.

  Fields:
    errors: Output only. Errors which caused the backfill job to fail.
    lastEndTime: Output only. Backfill job's end time.
    lastStartTime: Output only. Backfill job's start time.
    state: Output only. Backfill job state.
    trigger: Backfill job's triggering reason.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Backfill job state.

    Values:
      STATE_UNSPECIFIED: Default value.
      NOT_STARTED: Backfill job was never started for the stream object
        (stream has backfill strategy defined as manual or object was
        explicitly excluded from automatic backfill).
      PENDING: Backfill job will start pending available resources.
      ACTIVE: Backfill job is running.
      STOPPED: Backfill job stopped (next job run will start from beginning).
      FAILED: Backfill job failed (due to an error).
      COMPLETED: Backfill completed successfully.
      UNSUPPORTED: Backfill job failed since the table structure is currently
        unsupported for backfill.
    """
        STATE_UNSPECIFIED = 0
        NOT_STARTED = 1
        PENDING = 2
        ACTIVE = 3
        STOPPED = 4
        FAILED = 5
        COMPLETED = 6
        UNSUPPORTED = 7

    class TriggerValueValuesEnum(_messages.Enum):
        """Backfill job's triggering reason.

    Values:
      TRIGGER_UNSPECIFIED: Default value.
      AUTOMATIC: Object backfill job was triggered automatically according to
        the stream's backfill strategy.
      MANUAL: Object backfill job was triggered manually using the dedicated
        API.
    """
        TRIGGER_UNSPECIFIED = 0
        AUTOMATIC = 1
        MANUAL = 2
    errors = _messages.MessageField('Error', 1, repeated=True)
    lastEndTime = _messages.StringField(2)
    lastStartTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    trigger = _messages.EnumField('TriggerValueValuesEnum', 5)