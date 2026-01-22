from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class OverridesByRequestProtocolValue(_messages.Message):
    """The map between request protocol and the backend address.

    Messages:
      AdditionalProperty: An additional property for a
        OverridesByRequestProtocolValue object.

    Fields:
      additionalProperties: Additional properties of type
        OverridesByRequestProtocolValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a OverridesByRequestProtocolValue object.

      Fields:
        key: Name of the additional property.
        value: A BackendRule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BackendRule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)