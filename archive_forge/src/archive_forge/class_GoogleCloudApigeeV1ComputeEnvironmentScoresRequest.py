from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ComputeEnvironmentScoresRequest(_messages.Message):
    """Request for ComputeEnvironmentScores.

  Fields:
    filters: Optional. Filters are used to filter scored components. Return
      all the components if no filter is mentioned. Example: [{ "scorePath":
      "/org@myorg/envgroup@myenvgroup/env@myenv/proxies/proxy@myproxy/source"
      }, { "scorePath":
      "/org@myorg/envgroup@myenvgroup/env@myenv/proxies/proxy@myproxy/target",
      }] This will return components with path:
      "/org@myorg/envgroup@myenvgroup/env@myenv/proxies/proxy@myproxy/source"
      OR
      "/org@myorg/envgroup@myenvgroup/env@myenv/proxies/proxy@myproxy/target"
    pageSize: Optional. The maximum number of subcomponents to be returned in
      a single page. The service may return fewer than this value. If
      unspecified, at most 100 subcomponents will be returned in a single
      page.
    pageToken: Optional. A token that can be sent as `page_token` to retrieve
      the next page. If this field is omitted, there are no subsequent pages.
    timeRange: Required. Time range for score calculation. At most 14 days of
      scores will be returned.
  """
    filters = _messages.MessageField('GoogleCloudApigeeV1ComputeEnvironmentScoresRequestFilter', 1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    timeRange = _messages.MessageField('GoogleTypeInterval', 4)