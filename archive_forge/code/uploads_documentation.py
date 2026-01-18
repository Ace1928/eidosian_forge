from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import zip
Uploads path to Cloud Storage if it isn't already there.

  Translates local file system paths to Cloud Storage-style paths (i.e. using
  the Unix path separator '/').

  Args:
    path: str, the path to the directory. Can be a Cloud Storage ("gs://") path
      or a local filesystem path (no protocol).
    staging_bucket: storage_util.BucketReference or None. If the path is local,
      the bucket to which it should be uploaded.
    gs_prefix: str, prefix for the directory within the staging bucket.

  Returns:
    str, a Cloud Storage path where the directory has been uploaded (possibly
      prior to the execution of this function).

  Raises:
    MissingStagingBucketException: if `path` is a local path, but staging_bucket
      isn't found.
    BadDirectoryError: if the given directory couldn't be found or is empty.
  