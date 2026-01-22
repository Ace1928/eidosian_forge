from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsListRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsScanRunsListRequest object.

  Fields:
    pageSize: The maximum number of ScanRuns to return, can be limited by
      server. If not specified or not positive, the implementation will select
      a reasonable value.
    pageToken: A token identifying a page of results to be returned. This
      should be a `next_page_token` value returned from a previous List
      request. If unspecified, the first page of results is returned.
    parent: Required. The parent resource name, which should be a scan
      resource name in the format
      'projects/{projectId}/scanConfigs/{scanConfigId}'.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)