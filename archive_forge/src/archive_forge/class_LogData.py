from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
class LogData(object):
    """Representation of a log file.

  Stores information such as the name of the log file, its contents, and the
  command run.
  """
    TRACEBACK_MARKER = 'BEGIN CRASH STACKTRACE\n'
    COMMAND_REGEXP = 'Running \\[(gcloud(?:\\.[a-z-]+)*)\\]'

    def __init__(self, filename, command, contents, traceback):
        self.filename = filename
        self.command = command
        self.contents = contents
        self.traceback = traceback

    def __str__(self):
        crash_detected = ' (crash detected)' if self.traceback else ''
        return '[{0}]: [{1}]{2}'.format(self.relative_path, self.command, crash_detected)

    @property
    def relative_path(self):
        """Returns path of log file relative to log directory, or the full path.

    Returns the full path when the log file is not *in* the log directory.

    Returns:
      str, the relative or full path of log file.
    """
        logs_dir = config.Paths().logs_dir
        if logs_dir is None:
            return self.filename
        rel_path = os.path.relpath(self.filename, config.Paths().logs_dir)
        if rel_path.startswith(os.path.pardir + os.path.sep):
            return self.filename
        return rel_path

    @property
    def date(self):
        """Return the date that this log file was created, based on its filename.

    Returns:
      datetime.datetime that the log file was created or None, if the filename
        pattern was not recognized.
    """
        datetime_string = ':'.join(os.path.split(self.relative_path))
        datetime_format = log.DAY_DIR_FORMAT + ':' + log.FILENAME_FORMAT + log.LOG_FILE_EXTENSION
        try:
            return datetime.datetime.strptime(datetime_string, datetime_format)
        except ValueError:
            return None

    @classmethod
    def FromFile(cls, log_file):
        """Parse the file at the given path into a LogData.

    Args:
      log_file: str, the path to the log file to read

    Returns:
      LogData, representation of the log file
    """
        contents = file_utils.ReadFileContents(log_file)
        traceback = None
        command = None
        match = re.search(cls.COMMAND_REGEXP, contents)
        if match:
            dotted_cmd_string, = match.groups()
            command = ' '.join(dotted_cmd_string.split('.'))
        if cls.TRACEBACK_MARKER in contents:
            traceback = contents.split(cls.TRACEBACK_MARKER)[-1]
            traceback = re.split(log.LOG_PREFIX_PATTERN, traceback)[0]
            traceback = traceback.strip()
        return cls(log_file, command, contents, traceback)