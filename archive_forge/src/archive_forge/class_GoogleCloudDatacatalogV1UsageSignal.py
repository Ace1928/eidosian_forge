from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1UsageSignal(_messages.Message):
    """The set of all usage signals that Data Catalog stores. Note: Usually,
  these signals are updated daily. In rare cases, an update may fail but will
  be performed again on the next day.

  Messages:
    CommonUsageWithinTimeRangeValue: Common usage statistics over each of the
      predefined time ranges. Supported time ranges are `{"24H", "7D", "30D",
      "Lifetime"}`.
    UsageWithinTimeRangeValue: Output only. BigQuery usage statistics over
      each of the predefined time ranges. Supported time ranges are `{"24H",
      "7D", "30D"}`.

  Fields:
    commonUsageWithinTimeRange: Common usage statistics over each of the
      predefined time ranges. Supported time ranges are `{"24H", "7D", "30D",
      "Lifetime"}`.
    favoriteCount: Favorite count in the source system.
    updateTime: The end timestamp of the duration of usage statistics.
    usageWithinTimeRange: Output only. BigQuery usage statistics over each of
      the predefined time ranges. Supported time ranges are `{"24H", "7D",
      "30D"}`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CommonUsageWithinTimeRangeValue(_messages.Message):
        """Common usage statistics over each of the predefined time ranges.
    Supported time ranges are `{"24H", "7D", "30D", "Lifetime"}`.

    Messages:
      AdditionalProperty: An additional property for a
        CommonUsageWithinTimeRangeValue object.

    Fields:
      additionalProperties: Additional properties of type
        CommonUsageWithinTimeRangeValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CommonUsageWithinTimeRangeValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1CommonUsageStats attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1CommonUsageStats', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UsageWithinTimeRangeValue(_messages.Message):
        """Output only. BigQuery usage statistics over each of the predefined
    time ranges. Supported time ranges are `{"24H", "7D", "30D"}`.

    Messages:
      AdditionalProperty: An additional property for a
        UsageWithinTimeRangeValue object.

    Fields:
      additionalProperties: Additional properties of type
        UsageWithinTimeRangeValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UsageWithinTimeRangeValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1UsageStats attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1UsageStats', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    commonUsageWithinTimeRange = _messages.MessageField('CommonUsageWithinTimeRangeValue', 1)
    favoriteCount = _messages.IntegerField(2)
    updateTime = _messages.StringField(3)
    usageWithinTimeRange = _messages.MessageField('UsageWithinTimeRangeValue', 4)