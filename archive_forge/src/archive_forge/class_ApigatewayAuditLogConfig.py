from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayAuditLogConfig(_messages.Message):
    """Provides the configuration for logging a type of permissions. Example: {
  "audit_log_configs": [ { "log_type": "DATA_READ", "exempted_members": [
  "user:jose@example.com" ] }, { "log_type": "DATA_WRITE" } ] } This enables
  'DATA_READ' and 'DATA_WRITE' logging, while exempting jose@example.com from
  DATA_READ logging.

  Enums:
    LogTypeValueValuesEnum: The log type that this config enables.

  Fields:
    exemptedMembers: Specifies the identities that do not cause logging for
      this type of permission. Follows the same format of Binding.members.
    logType: The log type that this config enables.
  """

    class LogTypeValueValuesEnum(_messages.Enum):
        """The log type that this config enables.

    Values:
      LOG_TYPE_UNSPECIFIED: Default case. Should never be this.
      ADMIN_READ: Admin reads. Example: CloudIAM getIamPolicy
      DATA_WRITE: Data writes. Example: CloudSQL Users create
      DATA_READ: Data reads. Example: CloudSQL Users list
    """
        LOG_TYPE_UNSPECIFIED = 0
        ADMIN_READ = 1
        DATA_WRITE = 2
        DATA_READ = 3
    exemptedMembers = _messages.StringField(1, repeated=True)
    logType = _messages.EnumField('LogTypeValueValuesEnum', 2)