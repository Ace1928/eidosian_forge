from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _CleanupGCSStagingObjectsNotInUse(location, deployment_full_name, deployment_id):
    """Deletes staging object for all revisions except for last successful revision.

  Args:
    location: The location of deployment.
    deployment_full_name: string, the fully qualified name of the deployment,
      e.g. "projects/p/locations/l/deployments/d".
    deployment_id: the short name of the deployment.

  Raises:
    NotFoundError: If the bucket or folder does not exist.
  """
    gcs_client = storage_api.StorageClient()
    gcs_staging_dir = staging_bucket_util.DefaultGCSStagingDir(deployment_id, location)
    gcs_staging_dir_ref = resources.REGISTRY.Parse(gcs_staging_dir, collection='storage.objects')
    bucket_ref = storage_util.BucketReference(gcs_staging_dir_ref.bucket)
    staged_objects = set()
    try:
        items = gcs_client.ListBucket(bucket_ref, gcs_staging_dir_ref.object)
        for item in items:
            item_dir = '/'.join(item.name.split('/')[:4])
            staged_objects.add('gs://{0}/{1}'.format(gcs_staging_dir_ref.bucket, item_dir))
        if not staged_objects:
            return
    except storage_api.BucketNotFoundError:
        return
    op = configmanager_util.ListRevisions(deployment_full_name)
    revisions = sorted(op.revisions, key=lambda x: GetRevisionNumber(x.name), reverse=True)
    lastest_revision = revisions[0]
    if lastest_revision.terraformBlueprint is not None:
        staged_objects.discard(lastest_revision.terraformBlueprint.gcsSource)
    for revision in revisions:
        if str(revision.state) == 'APPLIED':
            if revision.terraformBlueprint is not None:
                staged_objects.discard(revision.terraformBlueprint.gcsSource)
            break
    for obj in staged_objects:
        staging_bucket_util.DeleteStagingGCSFolder(gcs_client, obj)