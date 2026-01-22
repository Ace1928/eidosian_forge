from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SummarizeSparkApplicationStagesResponse(_messages.Message):
    """Summary of a Spark Application stages.

  Fields:
    stagesSummary: Summary of a Spark Application Stages
  """
    stagesSummary = _messages.MessageField('StagesSummary', 1)