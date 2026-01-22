from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1EvaluationConfidenceLevelMetrics(_messages.Message):
    """Evaluations metrics, at a specific confidence level.

  Fields:
    confidenceLevel: The confidence level.
    metrics: The metrics at the specific confidence level.
  """
    confidenceLevel = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    metrics = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMetrics', 2)