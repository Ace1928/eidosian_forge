from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunAggregationQueryResponse(_messages.Message):
    """The response for Firestore.RunAggregationQuery.

  Fields:
    explainMetrics: Query explain metrics. This is only present when the
      RunAggregationQueryRequest.explain_options is provided, and it is sent
      only once with the last response in the stream.
    readTime: The time at which the aggregate result was computed. This is
      always monotonically increasing; in this case, the previous
      AggregationResult in the result stream are guaranteed not to have
      changed between their `read_time` and this one. If the query returns no
      results, a response with `read_time` and no `result` will be sent, and
      this represents the time at which the query was run.
    result: A single aggregation result. Not present when reporting partial
      progress.
    transaction: The transaction that was started as part of this request.
      Only present on the first response when the request requested to start a
      new transaction.
  """
    explainMetrics = _messages.MessageField('ExplainMetrics', 1)
    readTime = _messages.StringField(2)
    result = _messages.MessageField('AggregationResult', 3)
    transaction = _messages.BytesField(4)