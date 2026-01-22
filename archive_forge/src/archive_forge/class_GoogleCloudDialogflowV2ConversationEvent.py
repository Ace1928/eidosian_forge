from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationEvent(_messages.Message):
    """Represents a notification sent to Pub/Sub subscribers for conversation
  lifecycle events.

  Enums:
    TypeValueValuesEnum: The type of the event that this notification refers
      to.

  Fields:
    conversation: The unique identifier of the conversation this notification
      refers to. Format: `projects//conversations/`.
    errorStatus: More detailed information about an error. Only set for type
      UNRECOVERABLE_ERROR_IN_PHONE_CALL.
    newMessagePayload: Payload of NEW_MESSAGE event.
    type: The type of the event that this notification refers to.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the event that this notification refers to.

    Values:
      TYPE_UNSPECIFIED: Type not set.
      CONVERSATION_STARTED: A new conversation has been opened. This is fired
        when a telephone call is answered, or a conversation is created via
        the API.
      CONVERSATION_FINISHED: An existing conversation has closed. This is
        fired when a telephone call is terminated, or a conversation is closed
        via the API.
      HUMAN_INTERVENTION_NEEDED: An existing conversation has received
        notification from Dialogflow that human intervention is required.
      NEW_MESSAGE: An existing conversation has received a new message, either
        from API or telephony. It is configured in
        ConversationProfile.new_message_event_notification_config
      UNRECOVERABLE_ERROR: Unrecoverable error during a telephone call. In
        general non-recoverable errors only occur if something was
        misconfigured in the ConversationProfile corresponding to the call.
        After a non-recoverable error, Dialogflow may stop responding. We
        don't fire this event: * in an API call because we can directly return
        the error, or, * when we can recover from an error.
    """
        TYPE_UNSPECIFIED = 0
        CONVERSATION_STARTED = 1
        CONVERSATION_FINISHED = 2
        HUMAN_INTERVENTION_NEEDED = 3
        NEW_MESSAGE = 4
        UNRECOVERABLE_ERROR = 5
    conversation = _messages.StringField(1)
    errorStatus = _messages.MessageField('GoogleRpcStatus', 2)
    newMessagePayload = _messages.MessageField('GoogleCloudDialogflowV2Message', 3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)