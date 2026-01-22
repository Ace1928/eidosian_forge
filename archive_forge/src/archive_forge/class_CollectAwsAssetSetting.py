from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectAwsAssetSetting(_messages.Message):
    """The connection settings to collect asset data from AWS. This needs to be
  populated if connection type is COLLECT_AWS_ASSET. We have an option to
  enable scanning sensitive asset data. By default, the connection will both
  collect AWS asset metadata and scan sensitive asset data.

  Messages:
    QpsLimitValue: Optional. QPS rate limit for AWS API per each AWS service.
      For each entry, key is the name of AWS service and value is QPS rate
      limit.

  Fields:
    collectorRoleName: Required. Immutable. AWS collector role name. Collector
      role has delegate role as trusted entity, and is used to authenticate
      access AWS config data directly for each product.
    delegateAccountId: Required. AWS delegated account id. If this account id
      is in an AWS organization, we will attempt to discover all the AWS
      accounts in that AWS organization, which is referred to as AWS Account
      Auto Discovery feature. Note that: * This feature will be disabled when
      included_aws_account_ids is set. * This feature requires the
      delegate_role_name to be able to access [ListAccounts](https://docs.aws.
      amazon.com/organizations/latest/APIReference/API_ListAccounts.html).
    delegateRoleName: Required. Immutable. AWS delegate role name. GCP Service
      Account will assume a delegate role to get authenticated, then assume
      other collector roles to get authorized to collect config data. Delegate
      role ARN format -
      arn:aws:iam::{delegate_account_id}:role/{delegate_role_name}
    excludedAwsAccountIds: Optional. List of AWS accounts to exclude. This
      list should be mutually exclusive with included_aws_account_ids.
    includedAwsAccountIds: Optional. List of AWS accounts to collect data
      from. If this is provided, the AWS Account Auto Discovery will be
      disabled. This list should be mutually exclusive with
      excluded_aws_account_ids.
    qpsLimit: Optional. QPS rate limit for AWS API per each AWS service. For
      each entry, key is the name of AWS service and value is QPS rate limit.
    regionCodes: Optional. Region codes that this connection needs to collect
      data from, like `us-east-2`. If it's empty, then all regions should be
      used. Most AWS services and APIs are region specific. If region(s) is
      not specified, the data collection process can be very time consuming as
      all regions must be queried for all metadata.
    scanSensitiveDataSetting: Scan sensitive data setting.
    stsEndpointUri: Optional. AWS security token service endpoint. If a user
      disables the default global endpoint, user must provide regional
      endpoint to call for authentication.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class QpsLimitValue(_messages.Message):
        """Optional. QPS rate limit for AWS API per each AWS service. For each
    entry, key is the name of AWS service and value is QPS rate limit.

    Messages:
      AdditionalProperty: An additional property for a QpsLimitValue object.

    Fields:
      additionalProperties: Additional properties of type QpsLimitValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a QpsLimitValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    collectorRoleName = _messages.StringField(1)
    delegateAccountId = _messages.StringField(2)
    delegateRoleName = _messages.StringField(3)
    excludedAwsAccountIds = _messages.StringField(4, repeated=True)
    includedAwsAccountIds = _messages.StringField(5, repeated=True)
    qpsLimit = _messages.MessageField('QpsLimitValue', 6)
    regionCodes = _messages.StringField(7, repeated=True)
    scanSensitiveDataSetting = _messages.MessageField('ScanSensitiveDataSetting', 8)
    stsEndpointUri = _messages.StringField(9)