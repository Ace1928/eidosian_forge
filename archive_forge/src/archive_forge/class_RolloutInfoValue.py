from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RolloutInfoValue(_messages.Message):
    """Map for storing the information for all the rollout types for the
    appliance. The key for the map is the rollout_type_id, for example
    "appliance_rollout".

    Messages:
      AdditionalProperty: An additional property for a RolloutInfoValue
        object.

    Fields:
      additionalProperties: Additional properties of type RolloutInfoValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RolloutInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A RolloutInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('RolloutInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)