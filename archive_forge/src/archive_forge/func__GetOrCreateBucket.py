import os
import uuid
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def _GetOrCreateBucket(gcs_client, region):
    """Gets or Creates bucket used to store sources."""
    bucket = _GetBucketName(region)
    cors = [storage_util.GetMessages().Bucket.CorsValueListEntry(method=['GET'], origin=['https://*.cloud.google.com', 'https://*.corp.' + 'google.com', 'https://*.corp.' + 'google.com:*', 'https://*.cloud.google', 'https://*.byoid.goog'])]
    gcs_client.CreateBucketIfNotExists(bucket, location=region, check_ownership=True, cors=cors)
    return bucket