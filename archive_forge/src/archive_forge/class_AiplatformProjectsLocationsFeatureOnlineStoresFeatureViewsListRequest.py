from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsListRequest
  object.

  Fields:
    filter: Lists the FeatureViews that match the filter expression. The
      following filters are supported: * `create_time`: Supports `=`, `!=`,
      `<`, `>`, `>=`, and `<=` comparisons. Values must be in RFC 3339 format.
      * `update_time`: Supports `=`, `!=`, `<`, `>`, `>=`, and `<=`
      comparisons. Values must be in RFC 3339 format. * `labels`: Supports
      key-value equality as well as key presence. Examples: * `create_time >
      \\"2020-01-31T15:30:00.000000Z\\" OR update_time >
      \\"2020-01-31T15:30:00.000000Z\\"` --> FeatureViews created or updated
      after 2020-01-31T15:30:00.000000Z. * `labels.active = yes AND labels.env
      = prod` --> FeatureViews having both (active: yes) and (env: prod)
      labels. * `labels.env: *` --> Any FeatureView which has a label with
      'env' as the key.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `feature_view_id` * `create_time` * `update_time`
    pageSize: The maximum number of FeatureViews to return. The service may
      return fewer than this value. If unspecified, at most 1000 FeatureViews
      will be returned. The maximum value is 1000; any value greater than 1000
      will be coerced to 1000.
    pageToken: A page token, received from a previous
      FeatureOnlineStoreAdminService.ListFeatureViews call. Provide this to
      retrieve the subsequent page. When paginating, all other parameters
      provided to FeatureOnlineStoreAdminService.ListFeatureViews must match
      the call that provided the page token.
    parent: Required. The resource name of the FeatureOnlineStore to list
      FeatureViews. Format: `projects/{project}/locations/{location}/featureOn
      lineStores/{feature_online_store}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)