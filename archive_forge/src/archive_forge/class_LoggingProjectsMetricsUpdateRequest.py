from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingProjectsMetricsUpdateRequest(_messages.Message):
    """A LoggingProjectsMetricsUpdateRequest object.

  Fields:
    logMetric: A LogMetric resource to be passed as the request body.
    metricName: Required. The resource name of the metric to update:
      "projects/[PROJECT_ID]/metrics/[METRIC_ID]" The updated metric must be
      provided in the request and it's name field must be the same as
      [METRIC_ID] If the metric does not exist in [PROJECT_ID], then a new
      metric is created.
  """
    logMetric = _messages.MessageField('LogMetric', 1)
    metricName = _messages.StringField(2, required=True)