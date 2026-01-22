from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryDataLocalRequest(_messages.Message):
    """The request message for QueryDataLocal. This is identical to
  QueryDataRequest except for the associated resources.

  Fields:
    disableQueryCaching: Optional. If set to true, turns off all query caching
      on both the Log Analytics and BigQuery sides.
    parent: Required. The project in which the query will be run. The calling
      user must have the bigquery.jobs.create and bigquery.jobs.get
      permissions in this project.For example: "projects/PROJECT_ID"
    query: Required. The contents of the query.
    timeout: Optional. The timeout for the query. BigQuery will terminate the
      job if this duration is exceeded. If not set, the default is 5 minutes.
  """
    disableQueryCaching = _messages.BooleanField(1)
    parent = _messages.StringField(2)
    query = _messages.MessageField('AnalyticsQuery', 3)
    timeout = _messages.StringField(4)