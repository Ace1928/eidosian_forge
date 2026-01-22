from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _ShouldDeleteFile(self, now_seconds, path):
    """Determines if the file is old enough to be deleted.

    If the file is not a file that we recognize, return False.

    Args:
      now_seconds: int, The current time in seconds.
      path: str, The file or directory path to check.

    Returns:
      bool, True if it should be deleted, False otherwise.
    """
    if os.path.splitext(path)[1] not in _KNOWN_LOG_FILE_EXTENSIONS:
        return False
    stat_info = os.stat(path)
    return now_seconds - stat_info.st_mtime > self._GetMaxAge()