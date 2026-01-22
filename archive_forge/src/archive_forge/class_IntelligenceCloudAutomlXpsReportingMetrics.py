from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntelligenceCloudAutomlXpsReportingMetrics(_messages.Message):
    """A IntelligenceCloudAutomlXpsReportingMetrics object.

  Fields:
    effectiveTrainingDuration: The effective time training used. If set, this
      is used for quota management and billing. Deprecated. AutoML BE doesn't
      use this. Don't set.
    metricEntries: One entry per metric name. The values must be aggregated
      per metric name.
  """
    effectiveTrainingDuration = _messages.StringField(1)
    metricEntries = _messages.MessageField('IntelligenceCloudAutomlXpsMetricEntry', 2, repeated=True)