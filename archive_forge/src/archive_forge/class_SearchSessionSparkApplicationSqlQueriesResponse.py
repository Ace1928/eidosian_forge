from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSessionSparkApplicationSqlQueriesResponse(_messages.Message):
    """List of all queries for a Spark Application.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent
      SearchSessionSparkApplicationSqlQueriesRequest.
    sparkApplicationSqlQueries: Output only. SQL Execution Data
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationSqlQueries = _messages.MessageField('SqlExecutionUiData', 2, repeated=True)