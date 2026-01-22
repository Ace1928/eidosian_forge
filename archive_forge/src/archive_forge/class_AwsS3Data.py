from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsS3Data(_messages.Message):
    """An AwsS3Data resource can be a data source, but not a data sink. In an
  AwsS3Data resource, an object's name is the S3 object's key name.

  Fields:
    awsAccessKey: Input only. AWS access key used to sign the API requests to
      the AWS S3 bucket. Permissions on the bucket must be granted to the
      access ID of the AWS access key. For information on our data retention
      policy for user credentials, see [User credentials](/storage-
      transfer/docs/data-retention#user-credentials).
    bucketName: Required. S3 Bucket name (see [Creating a
      bucket](https://docs.aws.amazon.com/AmazonS3/latest/dev/create-bucket-
      get-location-example.html)).
    cloudfrontDomain: Optional. The CloudFront distribution domain name
      pointing to this bucket, to use when fetching. See [Transfer from S3 via
      CloudFront](https://cloud.google.com/storage-
      transfer/docs/s3-cloudfront) for more information. Format:
      `https://{id}.cloudfront.net` or any valid custom domain. Must begin
      with `https://`.
    credentialsSecret: Optional. The Resource name of a secret in Secret
      Manager. AWS credentials must be stored in Secret Manager in JSON
      format: { "access_key_id": "ACCESS_KEY_ID", "secret_access_key":
      "SECRET_ACCESS_KEY" } GoogleServiceAccount must be granted
      `roles/secretmanager.secretAccessor` for the resource. See [Configure
      access to a source: Amazon S3] (https://cloud.google.com/storage-
      transfer/docs/source-amazon-s3#secret_manager) for more information. If
      `credentials_secret` is specified, do not specify role_arn or
      aws_access_key. Format:
      `projects/{project_number}/secrets/{secret_name}`
    path: Root path to transfer objects. Must be an empty string or full path
      name that ends with a '/'. This field is treated as an object prefix. As
      such, it should generally not begin with a '/'.
    roleArn: The Amazon Resource Name (ARN) of the role to support temporary
      credentials via `AssumeRoleWithWebIdentity`. For more information about
      ARNs, see [IAM ARNs](https://docs.aws.amazon.com/IAM/latest/UserGuide/re
      ference_identifiers.html#identifiers-arns). When a role ARN is provided,
      Transfer Service fetches temporary credentials for the session using a
      `AssumeRoleWithWebIdentity` call for the provided role using the
      GoogleServiceAccount for this project.
  """
    awsAccessKey = _messages.MessageField('AwsAccessKey', 1)
    bucketName = _messages.StringField(2)
    cloudfrontDomain = _messages.StringField(3)
    credentialsSecret = _messages.StringField(4)
    path = _messages.StringField(5)
    roleArn = _messages.StringField(6)