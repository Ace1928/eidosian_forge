from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationResult(_messages.Message):
    """Validation result of the other-cloud connection.

  Enums:
    ConnectionStateValueValuesEnum: NOTE: Deprecated The state of the other-
      cloud connection

  Fields:
    cause: Optional. Some further information about the connection. When the
      connection does not pass the Delegated Role validation, including
      Delegated Role assumption and listing accounts when auto-discovery is
      enabled, it will contain the detailed failure reasons. If the Delegated
      Role validation passes, this field will always contain the validated
      Collector account number. In addition, when the connection state is
      AWS_INVALID_COLLECTOR_ACCOUNTS, it will provide the valid Collector Role
      rate, and the detailed reasons for all invalid accounts.
    connectionState: NOTE: Deprecated The state of the other-cloud connection
    connectionStatus: Required. The status of the other-cloud connection with
      one of the following values VALID: The connection has been set up at AWS
      properly: the GCP Service Agent can be properly assumed to the AWS
      delegated role, the AWS Delegated Role can be assumed to the Collector
      Role, and the AWS Collector Role has required permissions.
      AWS_FAILED_TO_ASSUME_DELEGATED_ROLE: The connection is invalid because
      the GCP service agent can not be properly assumed to an AWS delegated
      role. AWS_FAILED_TO_LIST_ACCOUNTS: The connection is invalid because the
      APS auto-discovery is enabled and the permission to allow the Delegated
      Role to list accounts in the organization has not been set properly.
      AWS_INVALID_COLLECTOR_ACCOUNTS: The connection has invalid Collector
      accounts. A predefined threshold of the maximum number of invalid
      Collector accounts will be defined. When the number of invalid Collector
      accounts exceeds this limit, the validation will stop. The reason for
      one Collector account's invalidity can be one of the following values.
      The detailed reason will be included in the cause field.
      AWS_FAILED_TO_ASSUME_COLLECTOR_ROLE: The Delegated Role can not be
      properly assumed to the AWS Collector Role in the account.
      AWS_COLLECTOR_ROLE_POLICY_MISSING_REQUIRED_PERMISSION: The Collector
      Role misses required policy settings.
    validationTime: Required. The time when the connection was validated.
  """

    class ConnectionStateValueValuesEnum(_messages.Enum):
        """NOTE: Deprecated The state of the other-cloud connection

    Values:
      UNKNOWN: Unknown.
      VALID: The connection has been set up at AWS properly: the GCP Service
        Agent can be properly assumed to the AWS delegated role and then the
        AWS Collector role with required permissions.
      FAILED_TO_ASSUME_DELEGATED_ROLE: The connection is invalid because the
        GCP service agent can not be properly assumed to an AWS delegated
        role.
      INVALID_FOR_OTHER_REASON: The connection setting is invalid for other
        reasons. The detailed cause is in the cause field.
    """
        UNKNOWN = 0
        VALID = 1
        FAILED_TO_ASSUME_DELEGATED_ROLE = 2
        INVALID_FOR_OTHER_REASON = 3
    cause = _messages.StringField(1)
    connectionState = _messages.EnumField('ConnectionStateValueValuesEnum', 2)
    connectionStatus = _messages.StringField(3)
    validationTime = _messages.StringField(4)