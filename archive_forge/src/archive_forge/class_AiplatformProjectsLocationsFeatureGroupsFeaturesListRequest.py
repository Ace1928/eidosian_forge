from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureGroupsFeaturesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureGroupsFeaturesListRequest object.

  Fields:
    filter: Lists the Features that match the filter expression. The following
      filters are supported: * `value_type`: Supports = and != comparisons. *
      `create_time`: Supports =, !=, <, >, >=, and <= comparisons. Values must
      be in RFC 3339 format. * `update_time`: Supports =, !=, <, >, >=, and <=
      comparisons. Values must be in RFC 3339 format. * `labels`: Supports
      key-value equality as well as key presence. Examples: * `value_type =
      DOUBLE` --> Features whose type is DOUBLE. * `create_time >
      \\"2020-01-31T15:30:00.000000Z\\" OR update_time >
      \\"2020-01-31T15:30:00.000000Z\\"` --> EntityTypes created or updated
      after 2020-01-31T15:30:00.000000Z. * `labels.active = yes AND labels.env
      = prod` --> Features having both (active: yes) and (env: prod) labels. *
      `labels.env: *` --> Any Feature which has a label with 'env' as the key.
    latestStatsCount: Only applicable for Vertex AI Feature Store (Legacy). If
      set, return the most recent ListFeaturesRequest.latest_stats_count of
      stats for each Feature in response. Valid value is [0, 10]. If number of
      stats exists < ListFeaturesRequest.latest_stats_count, return all
      existing stats.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `feature_id` * `value_type` (Not supported for FeatureRegistry Feature)
      * `create_time` * `update_time`
    pageSize: The maximum number of Features to return. The service may return
      fewer than this value. If unspecified, at most 1000 Features will be
      returned. The maximum value is 1000; any value greater than 1000 will be
      coerced to 1000.
    pageToken: A page token, received from a previous
      FeaturestoreService.ListFeatures call or
      FeatureRegistryService.ListFeatures call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      FeaturestoreService.ListFeatures or FeatureRegistryService.ListFeatures
      must match the call that provided the page token.
    parent: Required. The resource name of the Location to list Features.
      Format for entity_type as parent: `projects/{project}/locations/{locatio
      n}/featurestores/{featurestore}/entityTypes/{entity_type}` Format for
      feature_group as parent:
      `projects/{project}/locations/{location}/featureGroups/{feature_group}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    latestStatsCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)
    readMask = _messages.StringField(7)