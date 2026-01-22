from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def DeleteStagingGCSFolder(gcs_client, object_uri):
    """Deletes object if the object_uri is a staging path or else skips deletion.

  Args:
    gcs_client: a storage_api.StorageClient instance for interacting with GCS.
    object_uri: a gcs object path in format gs://path/to/gcs/object.

  Raises:
    NotFoundError: If the bucket or folder does not exist.
  """
    staging_dir_prefix = 'gs://{0}/{1}'.format(GetDefaultStagingBucket(), STAGING_DIR)
    if not object_uri.startswith(staging_dir_prefix):
        return
    gcs_staging_dir_ref = resources.REGISTRY.Parse(object_uri, collection='storage.objects')
    bucket_ref = storage_util.BucketReference(gcs_staging_dir_ref.bucket)
    try:
        items = gcs_client.ListBucket(bucket_ref, gcs_staging_dir_ref.object)
        for item in items:
            object_ref = storage_util.ObjectReference.FromName(gcs_staging_dir_ref.bucket, item.name)
            gcs_client.DeleteObject(object_ref)
    except storage_api.BucketNotFoundError:
        pass