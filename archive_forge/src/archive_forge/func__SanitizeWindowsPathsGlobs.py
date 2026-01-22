from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def _SanitizeWindowsPathsGlobs(filename_list, native_separator=os.sep):
    """Clean up path separators for globbing-resolved filenames.

  Python's globbing library resolves wildcards with OS-native path separators,
  however users could use POSIX paths even for configs in a Windows environment.
  This can result in multi-separator-character paths where /foo/bar/* will
  return a path match like /foo/bar\\\\baz.yaml.
  This function will make paths separators internally consistent.

  Args:
    filename_list: List of filenames resolved using python's glob library.
    native_separator: OS native path separator. Override for testing only.

  Returns:
    List of filenames edited to have consistent path separator characters.
  """
    if native_separator == POSIX_PATH_SEPARATOR:
        return filename_list
    sanitized_paths = []
    for f in filename_list:
        if POSIX_PATH_SEPARATOR in f:
            sanitized_paths.append(f.replace(native_separator, POSIX_PATH_SEPARATOR))
        else:
            sanitized_paths.append(f)
    return sanitized_paths