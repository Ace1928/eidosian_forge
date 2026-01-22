from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1BatchOperationMetadata(_messages.Message):
    """Metadata for the batch operations such as the current state. This is
  included in the `metadata` field of the `Operation` returned by the
  `GetOperation` call of the `google::longrunning::Operations` service.

  Enums:
    StateValueValuesEnum: The current state of the batch operation.

  Fields:
    endTime: The time when the batch request is finished and
      google.longrunning.Operation.done is set to true.
    state: The current state of the batch operation.
    submitTime: The time when the batch request was submitted to the server.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the batch operation.

    Values:
      STATE_UNSPECIFIED: Invalid.
      PROCESSING: Request is actively being processed.
      SUCCESSFUL: The request is done and at least one item has been
        successfully processed.
      FAILED: The request is done and no item has been successfully processed.
      CANCELLED: The request is done after the
        longrunning.Operations.CancelOperation has been called by the user.
        Any records that were processed before the cancel command are output
        as specified in the request.
    """
        STATE_UNSPECIFIED = 0
        PROCESSING = 1
        SUCCESSFUL = 2
        FAILED = 3
        CANCELLED = 4
    endTime = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    submitTime = _messages.StringField(3)