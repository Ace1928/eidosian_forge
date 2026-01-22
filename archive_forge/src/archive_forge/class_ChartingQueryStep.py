from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartingQueryStep(_messages.Message):
    """A query step defined as a set of charting configuration options. This
  may not be used as the first step in a query.

  Fields:
    breakdowns: Optional. The breakdowns for the measures of the chart. A
      breakdown turns a single measure into multiple effective measures, each
      one associated with a single value from the breakdown column.
    dimensions: Required. The dimension columns. How many dimensions to choose
      and how they're configured will depend on the chart type. A dimension is
      the labels for the data; e.g., the X axis for a line graph or the
      segment labels for a pie chart.
    measures: Required. The measures to be displayed within the chart. A
      measure is a data set to be displayed; e.g., a line on a line graph, a
      set of bars on a bar graph, or the segment widths on a pie chart.
    sorting: Optional. The top-level sorting that determines the order in
      which the results are returned.The column may be set to one of the
      dimension columns or left empty, which is equivalent. If no breakdowns
      are requested, it may be set to any measure column; if breakdowns are
      requested, sorting by measures is not supported. If there is an
      anonymous measure using aggregation "count", use the string "*" to name
      it here.
  """
    breakdowns = _messages.MessageField('ChartingBreakdown', 1, repeated=True)
    dimensions = _messages.MessageField('ChartingDimension', 2, repeated=True)
    measures = _messages.MessageField('ChartingMeasure', 3, repeated=True)
    sorting = _messages.MessageField('Sorting', 4)