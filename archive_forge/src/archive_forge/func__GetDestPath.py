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
def _GetDestPath(source_dirname, source_path, destination, dest_is_local):
    """Get dest path without the dirname of the source dir if present."""
    dest_path_suffix = source_path
    if source_path.startswith(source_dirname):
        dest_path_suffix = source_path[len(source_dirname):]
    if not dest_is_local:
        dest_path_suffix = dest_path_suffix.replace(os.path.sep, posixpath.sep)
    return _JoinPaths(destination, dest_path_suffix, gsutil_path=not dest_is_local)