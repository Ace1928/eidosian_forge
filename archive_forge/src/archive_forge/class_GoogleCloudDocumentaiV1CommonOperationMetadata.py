from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1CommonOperationMetadata(_messages.Message):
    """The common metadata for long running operations.

  Enums:
    StateValueValuesEnum: The state of the operation.

  Fields:
    createTime: The creation time of the operation.
    resource: A related resource to this operation.
    state: The state of the operation.
    stateMessage: A message providing more details about the current state of
      processing.
    updateTime: The last update time of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the operation.

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
    createTime = _messages.StringField(1)
    resource = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    stateMessage = _messages.StringField(4)
    updateTime = _messages.StringField(5)