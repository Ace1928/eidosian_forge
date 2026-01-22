from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsLocationPrometheusApiV1MetadataListRequest(_messages.Message):
    """A MonitoringProjectsLocationPrometheusApiV1MetadataListRequest object.

  Fields:
    limit: Maximum number of metrics to return.
    location: Location of the resource information. Has to be "global" for
      now.
    metric: The metric name for which to query metadata. If unset, all metric
      metadata is returned.
    name: Required. The workspace on which to execute the request. It is not
      part of the open source API but used as a request path prefix to
      distinguish different virtual Prometheus instances of Google Prometheus
      Engine. The format is: projects/PROJECT_ID_OR_NUMBER.
  """
    limit = _messages.IntegerField(1)
    location = _messages.StringField(2, required=True)
    metric = _messages.StringField(3)
    name = _messages.StringField(4, required=True)