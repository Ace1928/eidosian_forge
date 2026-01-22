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
def InstalledPaths(self):
    """Gets the list of files and dirs created by installing this component.

    Returns:
      list of str, The files and directories installed by this component.
    """
    with file_utils.FileReader(self.manifest_file) as f:
        files = [line.rstrip() for line in f]
    return files