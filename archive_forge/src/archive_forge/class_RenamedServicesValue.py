from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RenamedServicesValue(_messages.Message):
    """Map from original service names to renamed versions. This is used when
    the default generated types would cause a naming conflict. (Neither name
    is fully-qualified.) Example: Subscriber to SubscriberServiceApi.

    Messages:
      AdditionalProperty: An additional property for a RenamedServicesValue
        object.

    Fields:
      additionalProperties: Additional properties of type RenamedServicesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RenamedServicesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)