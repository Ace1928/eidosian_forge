from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NewJobStateValueValuesEnum(_messages.Enum):
    """The new job state.

    Values:
      STATE_UNSPECIFIED: Job state unspecified.
      QUEUED: Job is admitted (validated and persisted) and waiting for
        resources.
      SCHEDULED: Job is scheduled to run as soon as resource allocation is
        ready. The resource allocation may happen at a later time but with a
        high chance to succeed.
      RUNNING: Resource allocation has been successful. At least one Task in
        the Job is RUNNING.
      SUCCEEDED: All Tasks in the Job have finished successfully.
      FAILED: At least one Task in the Job has failed.
      DELETION_IN_PROGRESS: The Job will be deleted, but has not been deleted
        yet. Typically this is because resources used by the Job are still
        being cleaned up.
    """
    STATE_UNSPECIFIED = 0
    QUEUED = 1
    SCHEDULED = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    DELETION_IN_PROGRESS = 6