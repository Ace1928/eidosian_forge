from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultSetStats(_messages.Message):
    """Additional statistics about a ResultSet or PartialResultSet.

  Messages:
    QueryStatsValue: Aggregated statistics from the execution of the query.
      Only present when the query is profiled. For example, a query could
      return the statistics as follows: { "rows_returned": "3",
      "elapsed_time": "1.22 secs", "cpu_time": "1.19 secs" }

  Fields:
    queryPlan: QueryPlan for the query associated with this result.
    queryStats: Aggregated statistics from the execution of the query. Only
      present when the query is profiled. For example, a query could return
      the statistics as follows: { "rows_returned": "3", "elapsed_time": "1.22
      secs", "cpu_time": "1.19 secs" }
    rowCountExact: Standard DML returns an exact count of rows that were
      modified.
    rowCountLowerBound: Partitioned DML does not offer exactly-once semantics,
      so it returns a lower bound of the rows modified.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class QueryStatsValue(_messages.Message):
        """Aggregated statistics from the execution of the query. Only present
    when the query is profiled. For example, a query could return the
    statistics as follows: { "rows_returned": "3", "elapsed_time": "1.22
    secs", "cpu_time": "1.19 secs" }

    Messages:
      AdditionalProperty: An additional property for a QueryStatsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a QueryStatsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    queryPlan = _messages.MessageField('QueryPlan', 1)
    queryStats = _messages.MessageField('QueryStatsValue', 2)
    rowCountExact = _messages.IntegerField(3)
    rowCountLowerBound = _messages.IntegerField(4)