from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2CreateConversationModelEvaluationOperationMetadata(_messages.Message):
    """Metadata for a ConversationModels.CreateConversationModelEvaluation
  operation.

  Enums:
    StateValueValuesEnum: State of CreateConversationModel operation.

  Fields:
    conversationModel: The resource name of the conversation model. Format:
      `projects//locations//conversationModels/`
    conversationModelEvaluation: The resource name of the conversation model.
      Format: `projects//locations//conversationModels//evaluations/`
    createTime: Timestamp when the request to create conversation model was
      submitted. The time is measured on server side.
    state: State of CreateConversationModel operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of CreateConversationModel operation.

    Values:
      STATE_UNSPECIFIED: Operation status not specified.
      INITIALIZING: The operation is being prepared.
      RUNNING: The operation is running.
      CANCELLED: The operation is cancelled.
      SUCCEEDED: The operation has succeeded.
      FAILED: The operation has failed.
    """
        STATE_UNSPECIFIED = 0
        INITIALIZING = 1
        RUNNING = 2
        CANCELLED = 3
        SUCCEEDED = 4
        FAILED = 5
    conversationModel = _messages.StringField(1)
    conversationModelEvaluation = _messages.StringField(2)
    createTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)