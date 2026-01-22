from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2OperationMetadata(_messages.Message):
    """Contains metadata for the BatchProcessDocuments operation.

  Enums:
    StateValueValuesEnum: The state of the current batch processing.

  Fields:
    createTime: The creation time of the operation.
    state: The state of the current batch processing.
    stateMessage: A message providing more details about the current state of
      processing.
    updateTime: The last update time of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the current batch processing.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ACCEPTED: Request is received.
      WAITING: Request operation is waiting for scheduling.
      RUNNING: Request is being processed.
      SUCCEEDED: The batch processing completed successfully.
      CANCELLED: The batch processing was cancelled.
      FAILED: The batch processing has failed.
    """
        STATE_UNSPECIFIED = 0
        ACCEPTED = 1
        WAITING = 2
        RUNNING = 3
        SUCCEEDED = 4
        CANCELLED = 5
        FAILED = 6
    createTime = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    stateMessage = _messages.StringField(3)
    updateTime = _messages.StringField(4)