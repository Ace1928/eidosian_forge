from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SessionEventQueryDetail(_messages.Message):
    """Execution details of the query.

  Enums:
    EngineValueValuesEnum: Query Execution engine.

  Fields:
    dataProcessedBytes: The data processed by the query.
    duration: Time taken for execution of the query.
    engine: Query Execution engine.
    queryId: The unique Query id identifying the query.
    queryText: The query text executed.
    resultSizeBytes: The size of results the query produced.
  """

    class EngineValueValuesEnum(_messages.Enum):
        """Query Execution engine.

    Values:
      ENGINE_UNSPECIFIED: An unspecified Engine type.
      SPARK_SQL: Spark-sql engine is specified in Query.
      BIGQUERY: BigQuery engine is specified in Query.
    """
        ENGINE_UNSPECIFIED = 0
        SPARK_SQL = 1
        BIGQUERY = 2
    dataProcessedBytes = _messages.IntegerField(1)
    duration = _messages.StringField(2)
    engine = _messages.EnumField('EngineValueValuesEnum', 3)
    queryId = _messages.StringField(4)
    queryText = _messages.StringField(5)
    resultSizeBytes = _messages.IntegerField(6)