from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryPhase(_messages.Message):
    """RetryPhase contains the retry attempts and the metadata for initiating a
  new attempt.

  Enums:
    BackoffModeValueValuesEnum: Output only. The pattern of how the wait time
      of the retry attempt is calculated.

  Fields:
    attempts: Output only. Detail of a retry action.
    backoffMode: Output only. The pattern of how the wait time of the retry
      attempt is calculated.
    jobId: Output only. The job ID for the Job to retry.
    phaseId: Output only. The phase ID of the phase that includes the job
      being retried.
    totalAttempts: Output only. The number of attempts that have been made.
  """

    class BackoffModeValueValuesEnum(_messages.Enum):
        """Output only. The pattern of how the wait time of the retry attempt is
    calculated.

    Values:
      BACKOFF_MODE_UNSPECIFIED: No WaitMode is specified.
      BACKOFF_MODE_LINEAR: Increases the wait time linearly.
      BACKOFF_MODE_EXPONENTIAL: Increases the wait time exponentially.
    """
        BACKOFF_MODE_UNSPECIFIED = 0
        BACKOFF_MODE_LINEAR = 1
        BACKOFF_MODE_EXPONENTIAL = 2
    attempts = _messages.MessageField('RetryAttempt', 1, repeated=True)
    backoffMode = _messages.EnumField('BackoffModeValueValuesEnum', 2)
    jobId = _messages.StringField(3)
    phaseId = _messages.StringField(4)
    totalAttempts = _messages.IntegerField(5)