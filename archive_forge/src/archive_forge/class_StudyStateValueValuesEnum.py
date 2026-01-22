from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StudyStateValueValuesEnum(_messages.Enum):
    """The state of the Study.

    Values:
      STATE_UNSPECIFIED: The study state is unspecified.
      ACTIVE: The study is active.
      INACTIVE: The study is stopped due to an internal error.
      COMPLETED: The study is done when the service exhausts the parameter
        search space or max_trial_count is reached.
    """
    STATE_UNSPECIFIED = 0
    ACTIVE = 1
    INACTIVE = 2
    COMPLETED = 3