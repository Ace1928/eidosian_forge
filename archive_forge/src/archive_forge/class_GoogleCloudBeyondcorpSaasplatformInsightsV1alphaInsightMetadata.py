from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsightMetadata(_messages.Message):
    """Insight filters, groupings and aggregations that can be applied for the
  insight. Examples: aggregations, groups, field filters.

  Enums:
    AggregationsValueListEntryValuesEnum:

  Fields:
    aggregations: Output only. List of aggregation types available for
      insight.
    category: Output only. Category of the insight.
    displayName: Output only. Common name of the insight.
    fields: Output only. List of fields available for insight.
    groups: Output only. List of groupings available for insight.
    subCategory: Output only. Sub-Category of the insight.
    type: Output only. Type of the insight. It is metadata describing whether
      the insight is a metric (e.g. count) or a report (e.g. list, status).
  """

    class AggregationsValueListEntryValuesEnum(_messages.Enum):
        """AggregationsValueListEntryValuesEnum enum type.

    Values:
      AGGREGATION_UNSPECIFIED: Unspecified.
      HOURLY: Insight should be aggregated at hourly level.
      DAILY: Insight should be aggregated at daily level.
      WEEKLY: Insight should be aggregated at weekly level.
      MONTHLY: Insight should be aggregated at monthly level.
      CUSTOM_DATE_RANGE: Insight should be aggregated at the custom date range
        passed in as the start and end time in the request.
    """
        AGGREGATION_UNSPECIFIED = 0
        HOURLY = 1
        DAILY = 2
        WEEKLY = 3
        MONTHLY = 4
        CUSTOM_DATE_RANGE = 5
    aggregations = _messages.EnumField('AggregationsValueListEntryValuesEnum', 1, repeated=True)
    category = _messages.StringField(2)
    displayName = _messages.StringField(3)
    fields = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsightMetadataField', 4, repeated=True)
    groups = _messages.StringField(5, repeated=True)
    subCategory = _messages.StringField(6)
    type = _messages.StringField(7)