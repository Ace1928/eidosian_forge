from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatusValueValuesEnum(_messages.Enum):
    """The status code.

    Values:
      STATUS_UNSPECIFIED: Unspecifed code.
      DONE: The step has completed without errors.
      NOT_STARTED: The step has not started yet.
      IN_PROGRESS: The step is in progress.
      FAILED: The step has completed with errors.
    """
    STATUS_UNSPECIFIED = 0
    DONE = 1
    NOT_STARTED = 2
    IN_PROGRESS = 3
    FAILED = 4