from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import hashlib
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import times
from googlecloudsdk.third_party.appengine.tools import context_util
from six.moves import map  # pylint: disable=redefined-builtin
def _BuildFileUploadMap(manifest, source_dir, bucket_ref, tmp_dir, max_file_size):
    """Builds a map of files to upload, indexed by their hash.

  This skips already-uploaded files.

  Args:
    manifest: A dict containing the deployment manifest for a single service.
    source_dir: The relative source directory of the service.
    bucket_ref: The GCS bucket reference to upload files into.
    tmp_dir: The path to a temporary directory where generated files may be
      stored. If a file in the manifest is not found in the source directory,
      it will be retrieved from this directory instead.
    max_file_size: int, File size limit per individual file or None if no limit.

  Raises:
    LargeFileError: if one of the files to upload exceeds the maximum App Engine
    file size.

  Returns:
    A dict mapping hashes to file paths that should be uploaded.
  """
    files_to_upload = {}
    storage_client = storage_api.StorageClient()
    ttl = _GetLifecycleDeletePolicy(storage_client, bucket_ref)
    existing_items = set((o.name for o in storage_client.ListBucket(bucket_ref) if _IsTTLSafe(ttl, o)))
    skipped_size, total_size = (0, 0)
    for rel_path in manifest:
        full_path = os.path.join(source_dir, rel_path)
        if not os.path.exists(encoding.Encode(full_path, encoding='utf-8')):
            full_path = os.path.join(tmp_dir, rel_path)
        size = os.path.getsize(encoding.Encode(full_path, encoding='utf-8'))
        if max_file_size and size > max_file_size:
            raise LargeFileError(full_path, size, max_file_size)
        sha1_hash = manifest[rel_path]['sha1Sum']
        total_size += size
        if sha1_hash in existing_items:
            log.debug('Skipping upload of [{f}]'.format(f=rel_path))
            skipped_size += size
        else:
            files_to_upload[sha1_hash] = full_path
        if total_size:
            log.info('Incremental upload skipped {pct}% of data'.format(pct=round(100.0 * skipped_size / total_size, 2)))
    return files_to_upload