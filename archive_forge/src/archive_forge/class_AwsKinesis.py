from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AwsKinesis(_messages.Message):
    """Ingestion settings for Amazon Kinesis Data Streams.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates the
      state of the Kinesis ingestion source.

  Fields:
    awsRoleArn: Required. AWS role ARN to be used for Federated Identity
      authentication with Kinesis. Check the Pub/Sub docs for how to set up
      this role and the required permissions that need to be attached to it.
    consumerArn: Required. The Kinesis consumer ARN to used for ingestion in
      Enhanced Fan-Out mode. The consumer must be already created and ready to
      be used.
    gcpServiceAccount: Required. The GCP service account to be used for
      Federated Identity authentication with Kinesis (via a
      `AssumeRoleWithWebIdentity` call for the provided role). The
      `aws_role_arn` must be set up with `accounts.google.com:sub` equals to
      this service account number.
    state: Output only. An output-only field that indicates the state of the
      Kinesis ingestion source.
    streamArn: Required. The Kinesis stream ARN to ingest data from.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates the state of the
    Kinesis ingestion source.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: Ingestion is active.
      KINESIS_PERMISSION_DENIED: Permission denied encountered while consuming
        data from Kinesis. This can happen if: - The provided `aws_role_arn`
        does not exist or does not have the appropriate permissions attached.
        - The provided `aws_role_arn` is not set up properly for Identity
        Federation using `gcp_service_account`. - The Pub/Sub SA is not
        granted the `iam.serviceAccounts.getOpenIdToken` permission on
        `gcp_service_account`.
      PUBLISH_PERMISSION_DENIED: Permission denied encountered while
        publishing to the topic. This can happen if the Pub/Sub SA has not
        been granted the [appropriate publish
        permissions](https://cloud.google.com/pubsub/docs/access-
        control#pubsub.publisher)
      STREAM_NOT_FOUND: The Kinesis stream does not exist.
      CONSUMER_NOT_FOUND: The Kinesis consumer does not exist.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        KINESIS_PERMISSION_DENIED = 2
        PUBLISH_PERMISSION_DENIED = 3
        STREAM_NOT_FOUND = 4
        CONSUMER_NOT_FOUND = 5
    awsRoleArn = _messages.StringField(1)
    consumerArn = _messages.StringField(2)
    gcpServiceAccount = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    streamArn = _messages.StringField(5)