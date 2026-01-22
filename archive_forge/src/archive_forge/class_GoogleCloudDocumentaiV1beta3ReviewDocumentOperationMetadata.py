from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3ReviewDocumentOperationMetadata(_messages.Message):
    """The long-running operation metadata for the ReviewDocument method.

  Enums:
    StateValueValuesEnum: Used only when Operation.done is false.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    createTime: The creation time of the operation.
    questionId: The Crowd Compute question ID.
    state: Used only when Operation.done is false.
    stateMessage: A message providing more details about the current state of
      processing. For example, the error message if the operation is failed.
    updateTime: The last update time of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Used only when Operation.done is false.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      RUNNING: Operation is still running.
      CANCELLING: Operation is being cancelled.
      SUCCEEDED: Operation succeeded.
      FAILED: Operation failed.
      CANCELLED: Operation is cancelled.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        CANCELLING = 2
        SUCCEEDED = 3
        FAILED = 4
        CANCELLED = 5
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiV1beta3CommonOperationMetadata', 1)
    createTime = _messages.StringField(2)
    questionId = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stateMessage = _messages.StringField(5)
    updateTime = _messages.StringField(6)