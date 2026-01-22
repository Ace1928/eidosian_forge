from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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