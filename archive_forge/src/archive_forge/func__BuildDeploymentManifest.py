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
def _BuildDeploymentManifest(upload_dir, source_files, bucket_ref, tmp_dir):
    """Builds a deployment manifest for use with the App Engine Admin API.

  Args:
    upload_dir: str, path to the service's upload directory
    source_files: [str], relative paths to upload.
    bucket_ref: The reference to the bucket files will be placed in.
    tmp_dir: A temp directory for storing generated files (currently just source
        context files).
  Returns:
    A deployment manifest (dict) for use with the Admin API.
  """
    manifest = {}
    bucket_url = 'https://storage.googleapis.com/{0}'.format(bucket_ref.bucket)
    for rel_path in source_files:
        full_path = os.path.join(upload_dir, rel_path)
        sha1_hash = file_utils.Checksum.HashSingleFile(full_path, algorithm=hashlib.sha1)
        manifest_path = '/'.join([bucket_url, sha1_hash])
        manifest[_FormatForManifest(rel_path)] = {'sourceUrl': manifest_path, 'sha1Sum': sha1_hash}
    context_files = context_util.CreateContextFiles(tmp_dir, None, source_dir=upload_dir)
    for context_file in context_files:
        rel_path = os.path.basename(context_file)
        if rel_path in manifest:
            log.debug('Source context already exists. Using the existing file.')
            continue
        else:
            sha1_hash = file_utils.Checksum.HashSingleFile(context_file, algorithm=hashlib.sha1)
            manifest_path = '/'.join([bucket_url, sha1_hash])
            manifest[_FormatForManifest(rel_path)] = {'sourceUrl': manifest_path, 'sha1Sum': sha1_hash}
    return manifest