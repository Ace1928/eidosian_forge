from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryStepAggregation(_messages.Message):
    """An identifier for an aggregation. Aggregations are used for cases where
  we need to collapse a set of values into a single value, such as multiple
  points in a measure into a single bin.

  Fields:
    parameters: Optional. Parameters to be applied to the aggregation.
      Aggregations that support or require parameters are listed above.
    type: Required. The type of aggregation to apply. Legal values for this
      string are: "percentile" - Generates an APPROX_QUANTILES. Requires one
      integer or double parameter. Applies only to numeric values. Supports
      precision of up to 3 decimal places. "average" - Generates AVG().
      Applies only to numeric values. "count" - Generates COUNT(). "count-
      distinct" - Generates COUNT(DISTINCT). "count-distinct-approx" -
      Generates APPROX_COUNT_DISTINCT(). "max" - Generates MAX(). Applies only
      to numeric values. "min" - Generates MIN(). Applies only to numeric
      values. "sum" - Generates SUM(). Applies only to numeric values. "or" -
      Generates LOGICAL_OR(). Applies only to boolean values. "and" -
      Generates LOGICAL_AND(). Applies only to boolean values. "none", "" -
      Equivalent to no aggregation.
  """
    parameters = _messages.MessageField('Parameter', 1, repeated=True)
    type = _messages.StringField(2)