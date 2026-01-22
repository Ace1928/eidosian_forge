from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeSeriesFilterRatio(_messages.Message):
    """A pair of time series filters that define a ratio computation. The
  output time series is the pair-wise division of each aligned element from
  the numerator and denominator time series.

  Fields:
    denominator: The denominator of the ratio.
    numerator: The numerator of the ratio.
    pickTimeSeriesFilter: Ranking based time series filter.
    secondaryAggregation: Apply a second aggregation after the ratio is
      computed.
    statisticalTimeSeriesFilter: Statistics based time series filter. Note:
      This field is deprecated and completely ignored by the API.
  """
    denominator = _messages.MessageField('RatioPart', 1)
    numerator = _messages.MessageField('RatioPart', 2)
    pickTimeSeriesFilter = _messages.MessageField('PickTimeSeriesFilter', 3)
    secondaryAggregation = _messages.MessageField('Aggregation', 4)
    statisticalTimeSeriesFilter = _messages.MessageField('StatisticalTimeSeriesFilter', 5)