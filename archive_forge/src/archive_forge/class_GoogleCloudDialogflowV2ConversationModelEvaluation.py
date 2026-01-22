from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationModelEvaluation(_messages.Message):
    """Represents evaluation result of a conversation model.

  Fields:
    createTime: Output only. Creation time of this model.
    displayName: Optional. The display name of the model evaluation. At most
      64 bytes long.
    evaluationConfig: Optional. The configuration of the evaluation task.
    name: The resource name of the evaluation. Format:
      `projects//conversationModels//evaluations/`
    rawHumanEvalTemplateCsv: Output only. Human eval template in csv format.
      It tooks real-world conversations provided through input dataset,
      generates example suggestions for customer to verify quality of the
      model. For Smart Reply, the generated csv file contains columns of
      Context, (Suggestions,Q1,Q2)*3, Actual reply. Context contains at most
      10 latest messages in the conversation prior to the current suggestion.
      Q1: "Would you send it as the next message of agent?" Evaluated based on
      whether the suggest is appropriate to be sent by agent in current
      context. Q2: "Does the suggestion move the conversation closer to
      resolution?" Evaluated based on whether the suggestion provide
      solutions, or answers customer's question or collect information from
      customer to resolve the customer's issue. Actual reply column contains
      the actual agent reply sent in the context.
    smartReplyMetrics: Output only. Only available when model is for smart
      reply.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    evaluationConfig = _messages.MessageField('GoogleCloudDialogflowV2EvaluationConfig', 3)
    name = _messages.StringField(4)
    rawHumanEvalTemplateCsv = _messages.StringField(5)
    smartReplyMetrics = _messages.MessageField('GoogleCloudDialogflowV2SmartReplyMetrics', 6)