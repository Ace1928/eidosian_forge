from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsUptimeCheckConfigsListRequest(_messages.Message):
    """A MonitoringProjectsUptimeCheckConfigsListRequest object.

  Fields:
    filter: If provided, this field specifies the criteria that must be met by
      uptime checks to be included in the response.For more details, see
      Filtering syntax (https://cloud.google.com/monitoring/api/v3/sorting-
      and-filtering#filter_syntax).
    pageSize: The maximum number of results to return in a single response.
      The server may further constrain the maximum number of results returned
      in a single page. If the page_size is <=0, the server will decide the
      number of results to be returned.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return more results from the previous
      method call.
    parent: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) whose Uptime
      check configurations are listed. The format is:
      projects/[PROJECT_ID_OR_NUMBER]
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)