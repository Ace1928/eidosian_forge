from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalMigrateOrganizationMetadata(_messages.Message):
    """Long-running operation metadata message returned by the
  MigrateOrganization.

  Enums:
    OperationStateValueValuesEnum: Output only. Current operation state

  Fields:
    operationState: Output only. Current operation state
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """Output only. Current operation state

    Values:
      OPERATION_STATE_UNSPECIFIED: Unspecified.
      OPERATION_STATE_PENDING: Pending (Not started).
      OPERATION_STATE_RUNNING: In-progress.
      OPERATION_STATE_SUCCEEDED: Done successfully.
      OPERATION_STATE_FAILED: Done with errors.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        OPERATION_STATE_PENDING = 1
        OPERATION_STATE_RUNNING = 2
        OPERATION_STATE_SUCCEEDED = 3
        OPERATION_STATE_FAILED = 4
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 1)