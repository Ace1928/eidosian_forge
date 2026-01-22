from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumptionReport(_messages.Message):
    """ConsumptionReport is the report of ResourceAllowance consumptions in a
  time period.

  Messages:
    LatestPeriodConsumptionsValue: Output only. ResourceAllowance consumptions
      in the latest calendar period. Key is the calendar period in string
      format. Batch currently supports HOUR, DAY, MONTH and YEAR.

  Fields:
    latestPeriodConsumptions: Output only. ResourceAllowance consumptions in
      the latest calendar period. Key is the calendar period in string format.
      Batch currently supports HOUR, DAY, MONTH and YEAR.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LatestPeriodConsumptionsValue(_messages.Message):
        """Output only. ResourceAllowance consumptions in the latest calendar
    period. Key is the calendar period in string format. Batch currently
    supports HOUR, DAY, MONTH and YEAR.

    Messages:
      AdditionalProperty: An additional property for a
        LatestPeriodConsumptionsValue object.

    Fields:
      additionalProperties: Additional properties of type
        LatestPeriodConsumptionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LatestPeriodConsumptionsValue object.

      Fields:
        key: Name of the additional property.
        value: A PeriodConsumption attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PeriodConsumption', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    latestPeriodConsumptions = _messages.MessageField('LatestPeriodConsumptionsValue', 1)