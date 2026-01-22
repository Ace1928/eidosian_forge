from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryTimeSeriesStatsResponse(_messages.Message):
    """Represents security stats result as a collection of time series
  sequences.

  Fields:
    columns: Column names corresponding to the same order as the inner values
      in the stats field.
    nextPageToken: Next page token.
    values: Results of the query returned as a JSON array.
  """
    columns = _messages.StringField(1, repeated=True)
    nextPageToken = _messages.StringField(2)
    values = _messages.MessageField('GoogleCloudApigeeV1QueryTimeSeriesStatsResponseSequence', 3, repeated=True)