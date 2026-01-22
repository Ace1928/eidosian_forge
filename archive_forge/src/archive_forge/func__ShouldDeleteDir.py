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
def _ShouldDeleteDir(self, now, path):
    """Determines if the directory should be deleted.

    True iff:
    * path is a directory
    * path name is formatted according to DAY_DIR_FORMAT
    * age of path (according to DAY_DIR_FORMAT) is slightly older than the
      MAX_AGE of a log file

    Args:
      now: datetime.datetime object indicating the current date/time.
      path: the full path to the directory in question.

    Returns:
      bool, whether the path is a valid directory that should be deleted
    """
    if not os.path.isdir(path):
        return False
    try:
        dir_date = self._GetFileDatetime(path)
    except ValueError:
        return False
    dir_age = now - dir_date
    return dir_age > self._GetMaxAgeTimeDelta() + datetime.timedelta(1)