from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsFindingsListRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsScanRunsFindingsListRequest
  object.

  Fields:
    filter: Required. The filter expression. The expression must be in the
      format: . Supported field: 'finding_type'. Supported operator: '='.
    pageSize: The maximum number of Findings to return, can be limited by
      server. If not specified or not positive, the implementation will select
      a reasonable value.
    pageToken: A token identifying a page of results to be returned. This
      should be a `next_page_token` value returned from a previous List
      request. If unspecified, the first page of results is returned.
    parent: Required. The parent resource name, which should be a scan run
      resource name in the format
      'projects/{projectId}/scanConfigs/{scanConfigId}/scanRuns/{scanRunId}'.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)