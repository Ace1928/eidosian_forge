from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os.path
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _ExportStorageApi(gcs_bucket, source, destination):
    """Exports files and directories from an environment's GCS bucket."""
    old_source = source
    source = source.rstrip('*')
    object_is_subdir = old_source != source
    client = storage_api.StorageClient()
    source_ref = storage_util.ObjectReference.FromBucketRef(gcs_bucket, source)
    dest_is_local = True
    if destination.startswith('gs://'):
        destination = _JoinPaths(destination.strip(posixpath.sep), '', gsutil_path=True)
        dest_is_local = False
    elif not os.path.isdir(destination):
        raise command_util.Error('Destination for export must be a directory.')
    source_dirname = _JoinPaths(os.path.dirname(source), '', gsutil_path=True)
    to_export = _GetObjectOrSubdirObjects(source_ref, object_is_subdir=object_is_subdir, client=client)
    if dest_is_local:
        for obj in to_export:
            dest_path = _GetDestPath(source_dirname, obj.name, destination, True)
            files.MakeDir(os.path.dirname(dest_path))
            client.CopyFileFromGCS(obj, dest_path, overwrite=True)
    else:
        for obj in to_export:
            dest_object = storage_util.ObjectReference.FromUrl(_GetDestPath(source_dirname, obj.name, destination, False))
            client.Copy(obj, dest_object)