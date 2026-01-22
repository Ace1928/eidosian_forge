from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def ListBucket(self, bucket_ref, prefix=None):
    """Lists the contents of a cloud storage bucket.

    Args:
      bucket_ref: The reference to the bucket.
      prefix: str, Filter results to those whose names begin with this prefix.

    Yields:
      Object messages.

    Raises:
      BucketNotFoundError if the user-specified bucket does not exist.
      ListBucketError if there was an error listing the bucket.
    """
    request = self.messages.StorageObjectsListRequest(bucket=bucket_ref.bucket, prefix=prefix)
    try:
        for obj in list_pager.YieldFromList(self.client.objects, request, batch_size=None):
            yield obj
    except api_exceptions.HttpNotFoundError:
        raise BucketNotFoundError('Could not list bucket: [{bucket}] bucket does not exist.'.format(bucket=bucket_ref.bucket))
    except api_exceptions.HttpError as e:
        log.debug('Could not list bucket [{bucket}]: {e}'.format(bucket=bucket_ref.bucket, e=http_exc.HttpException(e)))
        raise ListBucketError('{code} Could not list bucket [{bucket}]: {message}'.format(code=e.status_code, bucket=bucket_ref.bucket, message=http_exc.HttpException(e, error_format='{status_message}')))