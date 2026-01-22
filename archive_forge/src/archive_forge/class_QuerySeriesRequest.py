from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuerySeriesRequest(_messages.Message):
    """QuerySeries holds all parameters of the Prometheus upstream API for
  querying series.

  Fields:
    end: The end time to evaluate the query for. Either floating point UNIX
      seconds or RFC3339 formatted timestamp.
    start: The start time to evaluate the query for. Either floating point
      UNIX seconds or RFC3339 formatted timestamp.
  """
    end = _messages.StringField(1)
    start = _messages.StringField(2)