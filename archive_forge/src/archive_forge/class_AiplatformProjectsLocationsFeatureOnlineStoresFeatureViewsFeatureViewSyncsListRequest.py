from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewS
  yncsListRequest object.

  Fields:
    filter: Lists the FeatureViewSyncs that match the filter expression. The
      following filters are supported: * `create_time`: Supports `=`, `!=`,
      `<`, `>`, `>=`, and `<=` comparisons. Values must be in RFC 3339 format.
      Examples: * `create_time > \\"2020-01-31T15:30:00.000000Z\\"` -->
      FeatureViewSyncs created after 2020-01-31T15:30:00.000000Z.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `create_time`
    pageSize: The maximum number of FeatureViewSyncs to return. The service
      may return fewer than this value. If unspecified, at most 1000
      FeatureViewSyncs will be returned. The maximum value is 1000; any value
      greater than 1000 will be coerced to 1000.
    pageToken: A page token, received from a previous
      FeatureOnlineStoreAdminService.ListFeatureViewSyncs call. Provide this
      to retrieve the subsequent page. When paginating, all other parameters
      provided to FeatureOnlineStoreAdminService.ListFeatureViewSyncs must
      match the call that provided the page token.
    parent: Required. The resource name of the FeatureView to list
      FeatureViewSyncs. Format: `projects/{project}/locations/{location}/featu
      reOnlineStores/{feature_online_store}/featureViews/{feature_view}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)