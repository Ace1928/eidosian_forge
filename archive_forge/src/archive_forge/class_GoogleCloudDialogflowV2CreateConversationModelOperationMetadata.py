from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2CreateConversationModelOperationMetadata(_messages.Message):
    """Metadata for a ConversationModels.CreateConversationModel operation.

  Enums:
    StateValueValuesEnum: State of CreateConversationModel operation.

  Fields:
    conversationModel: The resource name of the conversation model. Format:
      `projects//conversationModels/`
    createTime: Timestamp when the request to create conversation model is
      submitted. The time is measured on server side.
    state: State of CreateConversationModel operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of CreateConversationModel operation.

    Values:
      STATE_UNSPECIFIED: Invalid.
      PENDING: Request is submitted, but training has not started yet. The
        model may remain in this state until there is enough capacity to start
        training.
      SUCCEEDED: The training has succeeded.
      FAILED: The training has succeeded.
      CANCELLED: The training has been cancelled.
      CANCELLING: The training is in cancelling state.
      TRAINING: Custom model is training.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        SUCCEEDED = 2
        FAILED = 3
        CANCELLED = 4
        CANCELLING = 5
        TRAINING = 6
    conversationModel = _messages.StringField(1)
    createTime = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)