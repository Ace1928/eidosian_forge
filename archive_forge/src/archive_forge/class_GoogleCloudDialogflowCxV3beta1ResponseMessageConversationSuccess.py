from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ResponseMessageConversationSuccess(_messages.Message):
    """Indicates that the conversation succeeded, i.e., the bot handled the
  issue that the customer talked to it about. Dialogflow only uses this to
  determine which conversations should be counted as successful and doesn't
  process the metadata in this message in any way. Note that Dialogflow also
  considers conversations that get to the conversation end page as successful
  even if they don't return ConversationSuccess. You may set this, for
  example: * In the entry_fulfillment of a Page if entering the page indicates
  that the conversation succeeded. * In a webhook response when you determine
  that you handled the customer issue.

  Messages:
    MetadataValue: Custom metadata. Dialogflow doesn't impose any structure on
      this.

  Fields:
    metadata: Custom metadata. Dialogflow doesn't impose any structure on
      this.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Custom metadata. Dialogflow doesn't impose any structure on this.

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