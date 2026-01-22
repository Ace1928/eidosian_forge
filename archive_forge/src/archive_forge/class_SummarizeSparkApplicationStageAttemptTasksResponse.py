from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SummarizeSparkApplicationStageAttemptTasksResponse(_messages.Message):
    """Summary of tasks for a Spark Application stage attempt.

  Fields:
    stageAttemptTasksSummary: Summary of tasks for a Spark Application Stage
      Attempt
  """
    stageAttemptTasksSummary = _messages.MessageField('StageAttemptTasksSummary', 1)