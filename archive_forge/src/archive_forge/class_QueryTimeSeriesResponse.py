from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryTimeSeriesResponse(_messages.Message):
    """The QueryTimeSeries response.

  Fields:
    nextPageToken: If there are more results than have been returned, then
      this field is set to a non-empty value. To see the additional results,
      use that value as page_token in the next call to this method.
    partialErrors: Query execution errors that may have caused the time series
      data returned to be incomplete. The available data will be available in
      the response.
    timeSeriesData: The time series data.
    timeSeriesDescriptor: The descriptor for the time series data.
  """
    nextPageToken = _messages.StringField(1)
    partialErrors = _messages.MessageField('Status', 2, repeated=True)
    timeSeriesData = _messages.MessageField('TimeSeriesData', 3, repeated=True)
    timeSeriesDescriptor = _messages.MessageField('TimeSeriesDescriptor', 4)