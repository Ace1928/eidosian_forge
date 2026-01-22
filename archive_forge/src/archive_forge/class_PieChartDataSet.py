from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PieChartDataSet(_messages.Message):
    """Groups a time series query definition.

  Fields:
    dimensions: A dimension is a structured label, class, or category for a
      set of measurements in your data.
    measures: A measure is a measured value of a property in your data. For
      example, rainfall in inches, number of units sold, revenue gained, etc.
    minAlignmentPeriod: Optional. The lower bound on data point frequency for
      this data set, implemented by specifying the minimum alignment period to
      use in a time series query. For example, if the data is published once
      every 10 minutes, the min_alignment_period should be at least 10
      minutes. It would not make sense to fetch and align data at one minute
      intervals.
    sliceNameTemplate: Optional. A template for the name of the slice. This
      name will be displayed in the legend and the tooltip of the pie chart.
      It replaces the auto-generated names for the slices. For example, if the
      template is set to ${resource.labels.zone}, the zone's value will be
      used for the name instead of the default name.
    timeSeriesQuery: Required. The query for the PieChart. See,
      google.monitoring.dashboard.v1.TimeSeriesQuery.
  """
    dimensions = _messages.MessageField('Dimension', 1, repeated=True)
    measures = _messages.MessageField('Measure', 2, repeated=True)
    minAlignmentPeriod = _messages.StringField(3)
    sliceNameTemplate = _messages.StringField(4)
    timeSeriesQuery = _messages.MessageField('TimeSeriesQuery', 5)