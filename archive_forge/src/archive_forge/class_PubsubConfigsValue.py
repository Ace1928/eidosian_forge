from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PubsubConfigsValue(_messages.Message):
    """How this repository publishes a change in the repository through Cloud
    Pub/Sub. Keyed by the topic names.

    Messages:
      AdditionalProperty: An additional property for a PubsubConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type PubsubConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PubsubConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A PubsubConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PubsubConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)