from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def _FilesForSuffix(self, suffix):
    """Returns the files in the state directory that have the given suffix.

    Args:
      suffix: str, The file suffix to match on.

    Returns:
      list of str, The file names that match.
    """
    if not os.path.isdir(self._state_directory):
        return []
    files = os.listdir(self._state_directory)
    matching = [f for f in files if os.path.isfile(os.path.join(self._state_directory, f)) and f.endswith(suffix)]
    return matching