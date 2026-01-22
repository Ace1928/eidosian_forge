from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def ValidateBucketForCertificateAuthority(bucket_name):
    """Validates that a user-specified bucket can be used with a Private CA.

  Args:
    bucket_name: The name of the GCS bucket to validate.

  Returns:
    A BucketReference wrapping the given bucket name.

  Raises:
    InvalidArgumentException: when the given bucket can't be used with a CA.
  """
    messages = storage_util.GetMessages()
    client = storage_api.StorageClient(messages=messages)
    try:
        bucket = client.GetBucket(bucket_name, messages.StorageBucketsGetRequest.ProjectionValueValuesEnum.full)
        if not _BucketAllowsPublicObjectReads(bucket):
            log.warning('The specified bucket does not publicly expose new objects by default, so some clients may not be able to access the CA certificate or CRLs. For more details, see https://cloud.google.com/storage/docs/access-control/making-data-public')
        return storage_util.BucketReference(bucket_name)
    except storage_api.BucketNotFoundError:
        raise exceptions.InvalidArgumentException('gcs-bucket', 'The given bucket does not exist.')