from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrafficPercentStrategy(_messages.Message):
    """Strategy that specifies how clients of Google Service Controller want to
  send traffic to use different config versions. This is generally used by API
  proxy to split traffic based on your configured percentage for each config
  version. One example of how to gradually rollout a new service configuration
  using this strategy: Day 1 Rollout { id:
  "example.googleapis.com/rollout_20160206" traffic_percent_strategy {
  percentages: { "example.googleapis.com/20160201": 70.00
  "example.googleapis.com/20160206": 30.00 } } } Day 2 Rollout { id:
  "example.googleapis.com/rollout_20160207" traffic_percent_strategy: {
  percentages: { "example.googleapis.com/20160206": 100.00 } } }

  Messages:
    PercentagesValue: Maps service configuration IDs to their corresponding
      traffic percentage. Key is the service configuration ID, Value is the
      traffic percentage which must be greater than 0.0 and the sum must equal
      to 100.0.

  Fields:
    percentages: Maps service configuration IDs to their corresponding traffic
      percentage. Key is the service configuration ID, Value is the traffic
      percentage which must be greater than 0.0 and the sum must equal to
      100.0.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PercentagesValue(_messages.Message):
        """Maps service configuration IDs to their corresponding traffic
    percentage. Key is the service configuration ID, Value is the traffic
    percentage which must be greater than 0.0 and the sum must equal to 100.0.

    Messages:
      AdditionalProperty: An additional property for a PercentagesValue
        object.

    Fields:
      additionalProperties: Additional properties of type PercentagesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PercentagesValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
            key = _messages.StringField(1)
            value = _messages.FloatField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    percentages = _messages.MessageField('PercentagesValue', 1)