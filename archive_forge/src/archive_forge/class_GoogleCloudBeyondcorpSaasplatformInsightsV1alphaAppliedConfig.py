from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformInsightsV1alphaAppliedConfig(_messages.Message):
    """The configuration that was applied to generate the result.

  Enums:
    AggregationValueValuesEnum: Output only. Aggregation type applied.

  Fields:
    aggregation: Output only. Aggregation type applied.
    customGrouping: Output only. Customised grouping applied.
    endTime: Output only. Ending time for the duration for which insight was
      pulled.
    fieldFilter: Output only. Filters applied.
    group: Output only. Group id of the grouping applied.
    startTime: Output only. Starting time for the duration for which insight
      was pulled.
  """

    class AggregationValueValuesEnum(_messages.Enum):
        """Output only. Aggregation type applied.

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
    aggregation = _messages.EnumField('AggregationValueValuesEnum', 1)
    customGrouping = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformInsightsV1alphaCustomGrouping', 2)
    endTime = _messages.StringField(3)
    fieldFilter = _messages.StringField(4)
    group = _messages.StringField(5)
    startTime = _messages.StringField(6)