from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import heapq
import itertools
import os
import threading
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import regex_util
from googlecloudsdk.command_lib.storage import rsync_command_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
Initializes task.

    Args:
      container (Resource): Contains path of files to fetch.
      output_path (str): Where to write final sorted file list.
      exclude_pattern_strings (List[str]|None): Ignore resources whose paths
        matched these regex patterns.
      managed_folders_only (bool): If True, populates the file with managed
        folders. Otherwise, populates the file with object resources.
      ignore_symlinks (bool): Should FileWildcardIterator skip symlinks.
      recurse (bool): Gather nested items in container.
    