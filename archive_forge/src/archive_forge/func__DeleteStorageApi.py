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
def _DeleteStorageApi(gcs_bucket, target, gcs_subdir):
    """Deletes objects in a folder of an environment's bucket with storage API."""
    client = storage_api.StorageClient()
    delete_all = target == '*'
    target = '' if delete_all else target
    target_ref = storage_util.ObjectReference.FromBucketRef(gcs_bucket, _JoinPaths(gcs_subdir, target, gsutil_path=True))
    to_delete = _GetObjectOrSubdirObjects(target_ref, object_is_subdir=delete_all, client=client)
    for obj_ref in to_delete:
        client.DeleteObject(obj_ref)