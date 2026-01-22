from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetChange(_messages.Message):
    """Targets being watched have changed.

  Enums:
    TargetChangeTypeValueValuesEnum: The type of change that occurred.

  Fields:
    cause: The error that resulted in this change, if applicable.
    readTime: The consistent `read_time` for the given `target_ids` (omitted
      when the target_ids are not at a consistent snapshot). The stream is
      guaranteed to send a `read_time` with `target_ids` empty whenever the
      entire stream reaches a new consistent snapshot. ADD, CURRENT, and RESET
      messages are guaranteed to (eventually) result in a new consistent
      snapshot (while NO_CHANGE and REMOVE messages are not). For a given
      stream, `read_time` is guaranteed to be monotonically increasing.
    resumeToken: A token that can be used to resume the stream for the given
      `target_ids`, or all targets if `target_ids` is empty. Not set on every
      target change.
    targetChangeType: The type of change that occurred.
    targetIds: The target IDs of targets that have changed. If empty, the
      change applies to all targets. The order of the target IDs is not
      defined.
  """

    class TargetChangeTypeValueValuesEnum(_messages.Enum):
        """The type of change that occurred.

    Values:
      NO_CHANGE: No change has occurred. Used only to send an updated
        `resume_token`.
      ADD: The targets have been added.
      REMOVE: The targets have been removed.
      CURRENT: The targets reflect all changes committed before the targets
        were added to the stream. This will be sent after or with a
        `read_time` that is greater than or equal to the time at which the
        targets were added. Listeners can wait for this change if read-after-
        write semantics are desired.
      RESET: The targets have been reset, and a new initial state for the
        targets will be returned in subsequent changes. After the initial
        state is complete, `CURRENT` will be returned even if the target was
        previously indicated to be `CURRENT`.
    """
        NO_CHANGE = 0
        ADD = 1
        REMOVE = 2
        CURRENT = 3
        RESET = 4
    cause = _messages.MessageField('Status', 1)
    readTime = _messages.StringField(2)
    resumeToken = _messages.BytesField(3)
    targetChangeType = _messages.EnumField('TargetChangeTypeValueValuesEnum', 4)
    targetIds = _messages.IntegerField(5, repeated=True, variant=_messages.Variant.INT32)