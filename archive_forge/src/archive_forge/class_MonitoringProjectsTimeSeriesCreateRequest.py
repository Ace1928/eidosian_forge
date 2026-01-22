from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsTimeSeriesCreateRequest(_messages.Message):
    """A MonitoringProjectsTimeSeriesCreateRequest object.

  Fields:
    createTimeSeriesRequest: A CreateTimeSeriesRequest resource to be passed
      as the request body.
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER]
  """
    createTimeSeriesRequest = _messages.MessageField('CreateTimeSeriesRequest', 1)
    name = _messages.StringField(2, required=True)