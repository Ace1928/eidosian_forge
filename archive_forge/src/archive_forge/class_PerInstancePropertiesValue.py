from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PerInstancePropertiesValue(_messages.Message):
    """Per-instance properties to be set on individual instances. Keys of
    this map specify requested instance names. Can be empty if name_pattern is
    used.

    Messages:
      AdditionalProperty: An additional property for a
        PerInstancePropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        PerInstancePropertiesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PerInstancePropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A BulkInsertInstanceResourcePerInstanceProperties attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BulkInsertInstanceResourcePerInstanceProperties', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)