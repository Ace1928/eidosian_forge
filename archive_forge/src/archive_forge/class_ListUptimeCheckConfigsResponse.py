from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUptimeCheckConfigsResponse(_messages.Message):
    """The protocol for the ListUptimeCheckConfigs response.

  Fields:
    nextPageToken: This field represents the pagination token to retrieve the
      next page of results. If the value is empty, it means no further results
      for the request. To retrieve the next page of results, the value of the
      next_page_token is passed to the subsequent List method call (in the
      request message's page_token field).
    totalSize: The total number of Uptime check configurations for the
      project, irrespective of any pagination.
    uptimeCheckConfigs: The returned Uptime check configurations.
  """
    nextPageToken = _messages.StringField(1)
    totalSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    uptimeCheckConfigs = _messages.MessageField('UptimeCheckConfig', 3, repeated=True)