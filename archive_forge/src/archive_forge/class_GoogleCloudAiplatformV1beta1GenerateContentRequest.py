from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerateContentRequest(_messages.Message):
    """Request message for [PredictionService.GenerateContent].

  Fields:
    contents: Required. The content of the current conversation with the
      model. For single-turn queries, this is a single instance. For multi-
      turn queries, this is a repeated field that contains conversation
      history + latest request.
    generationConfig: Optional. Generation config.
    safetySettings: Optional. Per request settings for blocking unsafe
      content. Enforced on GenerateContentResponse.candidates.
    systemInstruction: Optional. The user provided system instructions for the
      model. Note: only text should be used in parts and content in each part
      will be in a separate paragraph.
    toolConfig: Optional. Tool config. This config is shared for all tools
      provided in the request.
    tools: Optional. A list of `Tools` the model may use to generate the next
      response. A `Tool` is a piece of code that enables the system to
      interact with external systems to perform an action, or set of actions,
      outside of knowledge and scope of the model.
  """
    contents = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 1, repeated=True)
    generationConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1GenerationConfig', 2)
    safetySettings = _messages.MessageField('GoogleCloudAiplatformV1beta1SafetySetting', 3, repeated=True)
    systemInstruction = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 4)
    toolConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolConfig', 5)
    tools = _messages.MessageField('GoogleCloudAiplatformV1beta1Tool', 6, repeated=True)