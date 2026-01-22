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
def _UploadSourceDirToGCS(gcs_client, source, gcs_source_staging, ignore_file):
    """Uploads a local directory to GCS.

  Uploads one file at a time rather than tarballing/zipping for compatibility
  with the back-end.

  Args:
    gcs_client: a storage_api.StorageClient instance for interacting with GCS.
    source: string, a path to a local directory.
    gcs_source_staging: resources.Resource, the bucket to upload to. This must
      already exist.
    ignore_file: optional string, a path to a gcloudignore file.
  """
    source_snapshot = deterministic_snapshot.DeterministicSnapshot(source, ignore_file=ignore_file)
    size_str = resource_transform.TransformSize(source_snapshot.uncompressed_size)
    log.status.Print('Uploading {num_files} file(s) totalling {size}.'.format(num_files=len(source_snapshot.files), size=size_str))
    for file_metadata in source_snapshot.GetSortedFiles():
        full_local_path = os.path.join(file_metadata.root, file_metadata.path)
        target_obj_ref = 'gs://{0}/{1}/{2}'.format(gcs_source_staging.bucket, gcs_source_staging.object, file_metadata.path)
        target_obj_ref = resources.REGISTRY.Parse(target_obj_ref, collection='storage.objects')
        gcs_client.CopyFileToGCS(full_local_path, target_obj_ref)