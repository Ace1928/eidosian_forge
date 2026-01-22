from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubstateValueValuesEnum(_messages.Enum):
    """Output only. Additional state information, which includes status
    reported by the agent.

    Values:
      UNSPECIFIED: The job substate is unknown.
      SUBMITTED: The Job is submitted to the agent.Applies to RUNNING state.
      QUEUED: The Job has been received and is awaiting execution (it might be
        waiting for a condition to be met). See the "details" field for the
        reason for the delay.Applies to RUNNING state.
      STALE_STATUS: The agent-reported status is out of date, which can be
        caused by a loss of communication between the agent and Dataproc. If
        the agent does not send a timely update, the job will fail.Applies to
        RUNNING state.
    """
    UNSPECIFIED = 0
    SUBMITTED = 1
    QUEUED = 2
    STALE_STATUS = 3