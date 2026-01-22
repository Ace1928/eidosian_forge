from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class MethodsValue(_messages.Message):
    """Methods on this resource.

    Messages:
      AdditionalProperty: An additional property for a MethodsValue object.

    Fields:
      additionalProperties: Description for any methods on this resource.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MethodsValue object.

      Fields:
        key: Name of the additional property.
        value: A RestMethod attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('RestMethod', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)