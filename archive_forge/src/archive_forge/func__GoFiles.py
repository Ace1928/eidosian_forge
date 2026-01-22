from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config as images_config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GoFiles(path):
    """Return list of '*.go' files under directory 'path'.

  Note that os.walk by default performs a top-down search, so files higher in
  the directory tree appear before others.

  Args:
    path: (str) Application path.

  Returns:
    ([str, ...]) List of full pathnames for all '*.go' files under 'path' dir.
  """
    go_files = []
    for root, _, filenames in os.walk(six.text_type(path)):
        for filename in fnmatch.filter(filenames, '*.go'):
            go_files.append(os.path.join(root, filename))
    return go_files