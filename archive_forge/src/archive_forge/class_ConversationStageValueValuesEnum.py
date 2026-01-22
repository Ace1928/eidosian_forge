from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConversationStageValueValuesEnum(_messages.Enum):
    """The stage of a conversation. It indicates whether the virtual agent or
    a human agent is handling the conversation. If the conversation is created
    with the conversation profile that has Dialogflow config set, defaults to
    ConversationStage.VIRTUAL_AGENT_STAGE; Otherwise, defaults to
    ConversationStage.HUMAN_ASSIST_STAGE. If the conversation is created with
    the conversation profile that has Dialogflow config set but explicitly
    sets conversation_stage to ConversationStage.HUMAN_ASSIST_STAGE, it skips
    ConversationStage.VIRTUAL_AGENT_STAGE stage and directly goes to
    ConversationStage.HUMAN_ASSIST_STAGE.

    Values:
      CONVERSATION_STAGE_UNSPECIFIED: Unknown. Should never be used after a
        conversation is successfully created.
      VIRTUAL_AGENT_STAGE: The conversation should return virtual agent
        responses into the conversation.
      HUMAN_ASSIST_STAGE: The conversation should not provide responses, just
        listen and provide suggestions.
    """
    CONVERSATION_STAGE_UNSPECIFIED = 0
    VIRTUAL_AGENT_STAGE = 1
    HUMAN_ASSIST_STAGE = 2