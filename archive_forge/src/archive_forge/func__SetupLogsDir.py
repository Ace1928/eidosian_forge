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
def _SetupLogsDir(self, logs_dir):
    """Creates the necessary log directories and get the file name to log to.

    Logs are created under the given directory.  There is a sub-directory for
    each day, and logs for individual invocations are created under that.

    Deletes files in this directory that are older than MAX_AGE.

    Args:
      logs_dir: str, Path to a directory to store log files under

    Returns:
      str, The path to the file to log to
    """
    now = datetime.datetime.now()
    day_dir_name = now.strftime(DAY_DIR_FORMAT)
    day_dir_path = os.path.join(logs_dir, day_dir_name)
    files.MakeDir(day_dir_path)
    filename = '{timestamp}{ext}'.format(timestamp=now.strftime(FILENAME_FORMAT), ext=LOG_FILE_EXTENSION)
    log_file = os.path.join(day_dir_path, filename)
    return log_file