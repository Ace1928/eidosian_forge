from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingProjectsMetricsCreateRequest(_messages.Message):
    """A LoggingProjectsMetricsCreateRequest object.

  Fields:
    logMetric: A LogMetric resource to be passed as the request body.
    parent: Required. The resource name of the project in which to create the
      metric: "projects/[PROJECT_ID]" The new metric must be provided in the
      request.
  """
    logMetric = _messages.MessageField('LogMetric', 1)
    parent = _messages.StringField(2, required=True)