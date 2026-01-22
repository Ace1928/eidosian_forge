from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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