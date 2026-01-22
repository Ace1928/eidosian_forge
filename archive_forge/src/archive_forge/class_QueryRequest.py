from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryRequest(_messages.Message):
    """A QueryRequest object.

  Fields:
    defaultDataset: [Optional] Specifies the default datasetId and projectId
      to assume for any unqualified table names in the query. If not set, all
      table names in the query string must be qualified in the format
      'datasetId.tableId'.
    dryRun: [Optional] If set to true, BigQuery doesn't run the job. Instead,
      if the query is valid, BigQuery returns statistics about the job such as
      how many bytes would be processed. If the query is invalid, an error
      returns. The default value is false.
    kind: The resource type of the request.
    maxResults: [Optional] The maximum number of rows of data to return per
      page of results. Setting this flag to a small value such as 1000 and
      then paging through results might improve reliability when the query
      result set is large. In addition to this limit, responses are also
      limited to 10 MB. By default, there is no maximum row count, and only
      the byte limit applies.
    preserveNulls: [Deprecated] This property is deprecated.
    query: [Required] A query string, following the BigQuery query syntax, of
      the query to execute. Example: "SELECT count(f1) FROM
      [myProjectId:myDatasetId.myTableId]".
    timeoutMs: [Optional] How long to wait for the query to complete, in
      milliseconds, before the request times out and returns. Note that this
      is only a timeout for the request, not the query. If the query takes
      longer to run than the timeout value, the call returns without any
      results and with the 'jobComplete' flag set to false. You can call
      GetQueryResults() to wait for the query to complete and read the
      results. The default value is 10000 milliseconds (10 seconds).
    useLegacySql: [Experimental] Specifies whether to use BigQuery's legacy
      SQL dialect for this query. The default value is true. If set to false,
      the query will use BigQuery's standard SQL:
      https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is
      set to false, the values of allowLargeResults and flattenResults are
      ignored; query will be run as if allowLargeResults is true and
      flattenResults is false.
    useQueryCache: [Optional] Whether to look for the result in the query
      cache. The query cache is a best-effort cache that will be flushed
      whenever tables in the query are modified. The default value is true.
  """
    defaultDataset = _messages.MessageField('DatasetReference', 1)
    dryRun = _messages.BooleanField(2)
    kind = _messages.StringField(3, default=u'bigquery#queryRequest')
    maxResults = _messages.IntegerField(4, variant=_messages.Variant.UINT32)
    preserveNulls = _messages.BooleanField(5)
    query = _messages.StringField(6)
    timeoutMs = _messages.IntegerField(7, variant=_messages.Variant.UINT32)
    useLegacySql = _messages.BooleanField(8)
    useQueryCache = _messages.BooleanField(9, default=True)