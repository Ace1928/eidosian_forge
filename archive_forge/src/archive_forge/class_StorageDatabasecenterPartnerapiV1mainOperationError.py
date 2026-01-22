from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainOperationError(_messages.Message):
    """An error that occurred during a backup creation operation.

  Enums:
    ErrorTypeValueValuesEnum:

  Fields:
    code: Identifies the specific error that occurred. REQUIRED
    errorType: A ErrorTypeValueValuesEnum attribute.
    message: Additional information about the error encountered. REQUIRED
  """

    class ErrorTypeValueValuesEnum(_messages.Enum):
        """ErrorTypeValueValuesEnum enum type.

    Values:
      OPERATION_ERROR_TYPE_UNSPECIFIED: UNSPECIFIED means product type is not
        known or available.
      KMS_KEY_ERROR: key destroyed, expired, not found, unreachable or
        permission denied.
      DATABASE_ERROR: Database is not accessible
      STOCKOUT_ERROR: The zone or region does not have sufficient resources to
        handle the request at the moment
      CANCELLATION_ERROR: User initiated cancellation
      SQLSERVER_ERROR: SQL server specific error
      INTERNAL_ERROR: Any other internal error.
    """
        OPERATION_ERROR_TYPE_UNSPECIFIED = 0
        KMS_KEY_ERROR = 1
        DATABASE_ERROR = 2
        STOCKOUT_ERROR = 3
        CANCELLATION_ERROR = 4
        SQLSERVER_ERROR = 5
        INTERNAL_ERROR = 6
    code = _messages.StringField(1)
    errorType = _messages.EnumField('ErrorTypeValueValuesEnum', 2)
    message = _messages.StringField(3)