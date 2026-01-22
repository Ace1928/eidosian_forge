from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceDeploymentStatus(_messages.Message):
    """Message decribing the status of a resource being deployed. Next tag: 7

  Enums:
    OperationValueValuesEnum: Operation to be performed on the resource .
    StateValueValuesEnum: Current status of the resource.

  Fields:
    errorMessage: The error details if the state is FAILED.
    errors: Output only. The error details if the state is FAILED.
    id: Output only. ID of the resource.
    name: Name of the resource.
    operation: Operation to be performed on the resource .
    state: Current status of the resource.
  """

    class OperationValueValuesEnum(_messages.Enum):
        """Operation to be performed on the resource .

    Values:
      OPERATION_UNSPECIFIED: Default value indicating the operation is
        unknown.
      APPLY: Apply configuration to resource.
      DESTROY: Destroy resource.
    """
        OPERATION_UNSPECIFIED = 0
        APPLY = 1
        DESTROY = 2

    class StateValueValuesEnum(_messages.Enum):
        """Current status of the resource.

    Values:
      STATE_UNSPECIFIED: Default value indicating the state is unknown.
      NOT_STARTED: Resource queued for deployment.
      RUNNING: Deployment in progress.
      FINISHED: Deployment completed.
      SUCCEEDED: Deployment completed successfully.
      FAILED: Deployment completed with failure.
    """
        STATE_UNSPECIFIED = 0
        NOT_STARTED = 1
        RUNNING = 2
        FINISHED = 3
        SUCCEEDED = 4
        FAILED = 5
    errorMessage = _messages.StringField(1)
    errors = _messages.MessageField('ResourceDeploymentError', 2, repeated=True)
    id = _messages.MessageField('ResourceID', 3)
    name = _messages.MessageField('TypedName', 4)
    operation = _messages.EnumField('OperationValueValuesEnum', 5)
    state = _messages.EnumField('StateValueValuesEnum', 6)