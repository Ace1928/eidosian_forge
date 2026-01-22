from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationQuery(_messages.Message):
    """Datastore query for running an aggregation over a Query.

  Fields:
    aggregations: Optional. Series of aggregations to apply over the results
      of the `nested_query`. Requires: * A minimum of one and maximum of five
      aggregations per query.
    nestedQuery: Nested query for aggregation
  """
    aggregations = _messages.MessageField('Aggregation', 1, repeated=True)
    nestedQuery = _messages.MessageField('Query', 2)