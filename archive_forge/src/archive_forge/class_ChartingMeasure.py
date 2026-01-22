from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChartingMeasure(_messages.Message):
    """A definition for a single measure column in the output table. Multiple
  measure columns will produce multiple curves, stacked bars, etc. depending
  on chart type.

  Fields:
    aggregation: Optional. The aggregation to apply to the input column.
      Required if binning is enabled on the dimension.
    column: Required. The column name within the output of the previous step
      to use. May be the same column as the dimension. May be left empty if
      the aggregation is set to "count" (but not "count-distinct" or "count-
      distinct-approx").
  """
    aggregation = _messages.MessageField('QueryStepAggregation', 1)
    column = _messages.StringField(2)