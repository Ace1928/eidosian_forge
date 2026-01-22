from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import io
import operator
import os
import tarfile
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def UploadSource(upload_dir, source_files, object_ref, gen_files=None):
    """Upload a gzipped tarball of the source directory to GCS.

  Note: To provide parity with docker's behavior, we must respect .dockerignore.

  Args:
    upload_dir: the directory to be archived.
    source_files: [str], relative paths to upload.
    object_ref: storage_util.ObjectReference, the Cloud Storage location to
      upload the source tarball to.
    gen_files: dict of filename to (str) contents of generated config and
      source context files.
  """
    gen_files = gen_files or {}
    dockerignore_contents = _GetDockerignoreExclusions(upload_dir, gen_files)
    included_paths = _GetIncludedPaths(upload_dir, source_files, dockerignore_contents)
    with files.TemporaryDirectory() as temp_dir:
        f = files.BinaryFileWriter(os.path.join(temp_dir, 'src.tgz'))
        with gzip.GzipFile(mode='wb', fileobj=f) as gz:
            _CreateTar(upload_dir, gen_files, included_paths, gz)
        f.close()
        storage_client = storage_api.StorageClient()
        storage_client.CopyFileToGCS(f.name, object_ref)