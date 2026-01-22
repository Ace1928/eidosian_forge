from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ScalingScheduleStatusValue(_messages.Message):
    """[Output Only] Status information of existing scaling schedules.

    Messages:
      AdditionalProperty: An additional property for a
        ScalingScheduleStatusValue object.

    Fields:
      additionalProperties: Additional properties of type
        ScalingScheduleStatusValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ScalingScheduleStatusValue object.

      Fields:
        key: Name of the additional property.
        value: A ScalingScheduleStatus attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ScalingScheduleStatus', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)