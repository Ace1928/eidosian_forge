from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplainMetrics(_messages.Message):
    """Explain metrics for the query.

  Fields:
    executionStats: Aggregated stats from the execution of the query. Only
      present when ExplainOptions.analyze is set to true.
    planSummary: Planning phase information for the query.
  """
    executionStats = _messages.MessageField('ExecutionStats', 1)
    planSummary = _messages.MessageField('PlanSummary', 2)