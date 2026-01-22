from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SummarizeSessionSparkApplicationExecutorsResponse(_messages.Message):
    """Consolidated summary of executors for a Spark Application.

  Fields:
    activeExecutorSummary: Consolidated summary for active executors.
    applicationId: Spark Application Id
    deadExecutorSummary: Consolidated summary for dead executors.
    totalExecutorSummary: Overall consolidated summary for all executors.
  """
    activeExecutorSummary = _messages.MessageField('ConsolidatedExecutorSummary', 1)
    applicationId = _messages.StringField(2)
    deadExecutorSummary = _messages.MessageField('ConsolidatedExecutorSummary', 3)
    totalExecutorSummary = _messages.MessageField('ConsolidatedExecutorSummary', 4)