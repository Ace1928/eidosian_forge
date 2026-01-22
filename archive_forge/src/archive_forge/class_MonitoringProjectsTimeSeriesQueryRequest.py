from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsTimeSeriesQueryRequest(_messages.Message):
    """A MonitoringProjectsTimeSeriesQueryRequest object.

  Fields:
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER]
    queryTimeSeriesRequest: A QueryTimeSeriesRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    queryTimeSeriesRequest = _messages.MessageField('QueryTimeSeriesRequest', 2)