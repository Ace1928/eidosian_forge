from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ResponseMessageLiveAgentHandoff(_messages.Message):
    """Indicates that the conversation should be handed off to a live agent.
  Dialogflow only uses this to determine which conversations were handed off
  to a human agent for measurement purposes. What else to do with this signal
  is up to you and your handoff procedures. You may set this, for example: *
  In the entry_fulfillment of a Page if entering the page indicates something
  went extremely wrong in the conversation. * In a webhook response when you
  determine that the customer issue can only be handled by a human.

  Messages:
    MetadataValue: Custom metadata for your handoff procedure. Dialogflow
      doesn't impose any structure on this.

  Fields:
    metadata: Custom metadata for your handoff procedure. Dialogflow doesn't
      impose any structure on this.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Custom metadata for your handoff procedure. Dialogflow doesn't impose
    any structure on this.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    metadata = _messages.MessageField('MetadataValue', 1)