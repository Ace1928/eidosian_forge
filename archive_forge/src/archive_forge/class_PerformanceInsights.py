from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerformanceInsights(_messages.Message):
    """Performance insights for the job.

  Fields:
    avgPreviousExecutionMs: Output only. Average execution ms of previous
      runs. Indicates the job ran slow compared to previous executions. To
      find previous executions, use INFORMATION_SCHEMA tables and filter jobs
      with same query hash.
    stagePerformanceChangeInsights: Output only. Query stage performance
      insights compared to previous runs, for diagnosing performance
      regression.
    stagePerformanceStandaloneInsights: Output only. Standalone query stage
      performance insights, for exploring potential improvements.
  """
    avgPreviousExecutionMs = _messages.IntegerField(1)
    stagePerformanceChangeInsights = _messages.MessageField('StagePerformanceChangeInsight', 2, repeated=True)
    stagePerformanceStandaloneInsights = _messages.MessageField('StagePerformanceStandaloneInsight', 3, repeated=True)