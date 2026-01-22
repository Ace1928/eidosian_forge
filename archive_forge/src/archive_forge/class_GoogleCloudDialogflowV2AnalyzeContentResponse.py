from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AnalyzeContentResponse(_messages.Message):
    """The response message for Participants.AnalyzeContent.

  Fields:
    automatedAgentReply: Only set if a Dialogflow automated agent has
      responded. Note that:
      AutomatedAgentReply.detect_intent_response.output_audio and
      AutomatedAgentReply.detect_intent_response.output_audio_config are
      always empty, use reply_audio instead.
    dtmfParameters: Indicates the parameters of DTMF.
    endUserSuggestionResults: The suggestions for end user. The order is the
      same as HumanAgentAssistantConfig.SuggestionConfig.feature_configs of
      HumanAgentAssistantConfig.end_user_suggestion_config. Same as
      human_agent_suggestion_results, any failure of Agent Assist features
      will not lead to the overall failure of an AnalyzeContent API call.
      Instead, the features will fail silently with the error field set in the
      corresponding SuggestionResult.
    humanAgentSuggestionResults: The suggestions for most recent human agent.
      The order is the same as
      HumanAgentAssistantConfig.SuggestionConfig.feature_configs of
      HumanAgentAssistantConfig.human_agent_suggestion_config. Note that any
      failure of Agent Assist features will not lead to the overall failure of
      an AnalyzeContent API call. Instead, the features will fail silently
      with the error field set in the corresponding SuggestionResult.
    message: Message analyzed by CCAI.
    replyAudio: The audio data bytes encoded as specified in the request. This
      field is set if: - `reply_audio_config` was specified in the request, or
      - The automated agent responded with audio to play to the user. In such
      case, `reply_audio.config` contains settings used to synthesize the
      speech. In some scenarios, multiple output audio fields may be present
      in the response structure. In these cases, only the top-most-level audio
      output has content.
    replyText: The output text content. This field is set if the automated
      agent responded with text to show to the user.
  """
    automatedAgentReply = _messages.MessageField('GoogleCloudDialogflowV2AutomatedAgentReply', 1)
    dtmfParameters = _messages.MessageField('GoogleCloudDialogflowV2DtmfParameters', 2)
    endUserSuggestionResults = _messages.MessageField('GoogleCloudDialogflowV2SuggestionResult', 3, repeated=True)
    humanAgentSuggestionResults = _messages.MessageField('GoogleCloudDialogflowV2SuggestionResult', 4, repeated=True)
    message = _messages.MessageField('GoogleCloudDialogflowV2Message', 5)
    replyAudio = _messages.MessageField('GoogleCloudDialogflowV2OutputAudio', 6)
    replyText = _messages.StringField(7)