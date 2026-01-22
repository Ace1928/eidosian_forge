from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionStats(_messages.Message):
    """Execution statistics for the query.

  Messages:
    DebugStatsValue: Debugging statistics from the execution of the query.
      Note that the debugging stats are subject to change as Firestore
      evolves. It could include: { "indexes_entries_scanned": "1000",
      "documents_scanned": "20", "billing_details" : { "documents_billable":
      "20", "index_entries_billable": "1000", "min_query_cost": "0" } }

  Fields:
    debugStats: Debugging statistics from the execution of the query. Note
      that the debugging stats are subject to change as Firestore evolves. It
      could include: { "indexes_entries_scanned": "1000", "documents_scanned":
      "20", "billing_details" : { "documents_billable": "20",
      "index_entries_billable": "1000", "min_query_cost": "0" } }
    executionDuration: Total time to execute the query in the backend.
    readOperations: Total billable read operations.
    resultsReturned: Total number of results returned, including documents,
      projections, aggregation results, keys.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DebugStatsValue(_messages.Message):
        """Debugging statistics from the execution of the query. Note that the
    debugging stats are subject to change as Firestore evolves. It could
    include: { "indexes_entries_scanned": "1000", "documents_scanned": "20",
    "billing_details" : { "documents_billable": "20",
    "index_entries_billable": "1000", "min_query_cost": "0" } }

    Messages:
      AdditionalProperty: An additional property for a DebugStatsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DebugStatsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    debugStats = _messages.MessageField('DebugStatsValue', 1)
    executionDuration = _messages.StringField(2)
    readOperations = _messages.IntegerField(3)
    resultsReturned = _messages.IntegerField(4)