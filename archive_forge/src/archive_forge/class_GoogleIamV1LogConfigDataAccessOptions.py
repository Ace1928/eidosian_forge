from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1LogConfigDataAccessOptions(_messages.Message):
    """Write a Data Access (Gin) log

  Enums:
    LogModeValueValuesEnum:

  Fields:
    logMode: A LogModeValueValuesEnum attribute.
  """

    class LogModeValueValuesEnum(_messages.Enum):
        """LogModeValueValuesEnum enum type.

    Values:
      LOG_MODE_UNSPECIFIED: Client is not required to write a partial Gin log
        immediately after the authorization check. If client chooses to write
        one and it fails, client may either fail open (allow the operation to
        continue) or fail closed (handle as a DENY outcome).
      LOG_FAIL_CLOSED: The application's operation in the context of which
        this authorization check is being made may only be performed if it is
        successfully logged to Gin. For instance, the authorization library
        may satisfy this obligation by emitting a partial log entry at
        authorization check time and only returning ALLOW to the application
        if it succeeds. If a matching Rule has this directive, but the client
        has not indicated that it will honor such requirements, then the IAM
        check will result in authorization failure by setting
        CheckPolicyResponse.success=false.
    """
        LOG_MODE_UNSPECIFIED = 0
        LOG_FAIL_CLOSED = 1
    logMode = _messages.EnumField('LogModeValueValuesEnum', 1)