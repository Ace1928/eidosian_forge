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
class _LogManager(object):
    """A class to manage the logging handlers based on how calliope is being used.

  We want to always log to a file, in addition to logging to stdout if in CLI
  mode.  This sets up the required handlers to do this.
  """
    FILE_ONLY_LOGGER_NAME = '___FILE_ONLY___'

    def __init__(self):
        self._file_formatter = _LogFileFormatter()
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(logging.NOTSET)
        self.file_only_logger = logging.getLogger(_LogManager.FILE_ONLY_LOGGER_NAME)
        self.file_only_logger.setLevel(logging.NOTSET)
        self.file_only_logger.propagate = False
        self._logs_dirs = []
        self._console_formatter = None
        self._user_output_filter = _UserOutputFilter(DEFAULT_USER_OUTPUT_ENABLED)
        self.stdout_stream_wrapper = _StreamWrapper(None)
        self.stderr_stream_wrapper = _StreamWrapper(None)
        self.stdout_writer = _ConsoleWriter(self.file_only_logger, self._user_output_filter, self.stdout_stream_wrapper)
        self.stderr_writer = _ConsoleWriter(self.file_only_logger, self._user_output_filter, self.stderr_stream_wrapper, always_flush=True)
        self.verbosity = None
        self.user_output_enabled = None
        self.current_log_file = None
        self.Reset(sys.stdout, sys.stderr)

    def Reset(self, stdout, stderr):
        """Resets all logging functionality to its default state."""
        self._root_logger.handlers[:] = []
        self.stdout_stream_wrapper.stream = stdout
        self.stderr_stream_wrapper.stream = stderr
        json_formatter = _JsonFormatter(REQUIRED_STRUCTURED_RECORD_FIELDS)
        std_console_formatter = _ConsoleFormatter(stderr)
        console_formatter = _ConsoleLoggingFormatterMuxer(json_formatter, self.stderr_writer, default_formatter=std_console_formatter)
        self._console_formatter = console_formatter
        self.stderr_handler = logging.StreamHandler(stderr)
        self.stderr_handler.setFormatter(self._console_formatter)
        self.stderr_handler.setLevel(DEFAULT_VERBOSITY)
        self._root_logger.addHandler(self.stderr_handler)
        for f in self.file_only_logger.handlers:
            f.close()
        self.file_only_logger.handlers[:] = []
        self.file_only_logger.addHandler(_NullHandler())
        self.file_only_logger.setLevel(logging.NOTSET)
        self.SetVerbosity(None)
        self.SetUserOutputEnabled(None)
        self.current_log_file = None
        logging.getLogger('urllib3.connectionpool').addFilter(NoHeaderErrorFilter())

    def SetVerbosity(self, verbosity):
        """Sets the active verbosity for the logger.

    Args:
      verbosity: int, A verbosity constant from the logging module that
        determines what level of logs will show in the console. If None, the
        value from properties or the default will be used.

    Returns:
      int, The current verbosity.
    """
        if verbosity is None:
            verbosity_string = properties.VALUES.core.verbosity.Get()
            if verbosity_string is not None:
                verbosity = VALID_VERBOSITY_STRINGS.get(verbosity_string.lower())
        if verbosity is None:
            verbosity = DEFAULT_VERBOSITY
        if self.verbosity == verbosity:
            return self.verbosity
        self.stderr_handler.setLevel(verbosity)
        old_verbosity = self.verbosity
        self.verbosity = verbosity
        return old_verbosity

    def SetUserOutputEnabled(self, enabled):
        """Sets whether user output should go to the console.

    Args:
      enabled: bool, True to enable output, False to suppress.  If None, the
        value from properties or the default will be used.

    Returns:
      bool, The old value of enabled.
    """
        if enabled is None:
            enabled = properties.VALUES.core.user_output_enabled.GetBool(validate=False)
        if enabled is None:
            enabled = DEFAULT_USER_OUTPUT_ENABLED
        self._user_output_filter.enabled = enabled
        old_enabled = self.user_output_enabled
        self.user_output_enabled = enabled
        return old_enabled

    def _GetMaxLogDays(self):
        """Gets the max log days for the logger.

    Returns:
      max_log_days: int, the maximum days for log file retention
    """
        return properties.VALUES.core.max_log_days.GetInt()

    def _GetMaxAge(self):
        """Gets max_log_day's worth of seconds."""
        return 60 * 60 * 24 * self._GetMaxLogDays()

    def _GetMaxAgeTimeDelta(self):
        return datetime.timedelta(days=self._GetMaxLogDays())

    def _GetFileDatetime(self, path):
        return datetime.datetime.strptime(os.path.basename(path), DAY_DIR_FORMAT)

    def AddLogsDir(self, logs_dir):
        """Adds a new logging directory and configures file logging.

    Args:
      logs_dir: str, Path to a directory to store log files under.  This method
        has no effect if this is None, or if this directory has already been
        registered.
    """
        if not logs_dir or logs_dir in self._logs_dirs:
            return
        self._logs_dirs.append(logs_dir)
        self._CleanUpLogs(logs_dir)
        if properties.VALUES.core.disable_file_logging.GetBool():
            return
        try:
            log_file = self._SetupLogsDir(logs_dir)
            file_handler = logging.FileHandler(log_file, encoding=LOG_FILE_ENCODING)
        except (OSError, IOError, files.Error) as exp:
            warning('Could not setup log file in {0}, ({1}: {2}.\nThe configuration directory may not be writable. To learn more, see https://cloud.google.com/sdk/docs/configurations#creating_a_configuration'.format(logs_dir, type(exp).__name__, exp))
            return
        self.current_log_file = log_file
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(self._file_formatter)
        self._root_logger.addHandler(file_handler)
        self.file_only_logger.addHandler(file_handler)

    def _CleanUpLogs(self, logs_dir):
        """Clean up old log files if log cleanup has been enabled."""
        if self._GetMaxLogDays():
            try:
                self._CleanLogsDir(logs_dir)
            except OSError:
                pass

    def _CleanLogsDir(self, logs_dir):
        """Cleans up old log files form the given logs directory.

    Args:
      logs_dir: str, The path to the logs directory.
    """
        now = datetime.datetime.now()
        now_seconds = time.time()
        try:
            dirnames = os.listdir(logs_dir)
        except (OSError, UnicodeError):
            return
        for dirname in dirnames:
            dir_path = os.path.join(logs_dir, dirname)
            if self._ShouldDeleteDir(now, dir_path):
                for filename in os.listdir(dir_path):
                    log_file_path = os.path.join(dir_path, filename)
                    if self._ShouldDeleteFile(now_seconds, log_file_path):
                        os.remove(log_file_path)
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass

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