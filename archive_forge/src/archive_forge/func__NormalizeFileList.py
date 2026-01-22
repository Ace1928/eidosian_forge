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
def _NormalizeFileList(file_list):
    """Removes non-empty directory entries and sorts resulting list."""
    parent_directories = set([])
    directories = set([])
    files = set([])
    for f in file_list:
        norm_file_path = posixpath.normpath(f)
        if f.endswith('/'):
            directories.add(norm_file_path + '/')
        else:
            files.add(norm_file_path)
        norm_file_path = os.path.dirname(norm_file_path)
        while norm_file_path:
            parent_directories.add(norm_file_path + '/')
            norm_file_path = os.path.dirname(norm_file_path)
    return sorted(directories - parent_directories | files)