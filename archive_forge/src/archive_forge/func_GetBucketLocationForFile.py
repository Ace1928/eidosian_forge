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
def GetBucketLocationForFile(self, object_path):
    """Returns the location of the bucket for a file.

    Args:
      object_path: str, the path of the file in GCS.

    Returns:
      str, bucket location (region) for given object in GCS.

    Raises:
      BucketNotFoundError if bucket from the object path is not found.
    """
    object_reference = storage_util.ObjectReference.FromUrl(object_path)
    bucket_name = object_reference.bucket
    get_bucket_req = self.messages.StorageBucketsGetRequest(bucket=bucket_name)
    try:
        source_bucket = self.client.buckets.Get(get_bucket_req)
        return source_bucket.location
    except api_exceptions.HttpNotFoundError:
        raise BucketNotFoundError('Could not get location for file: [{bucket}] bucket does not exist.'.format(bucket=bucket_name))