from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import getpass
import io
import itertools
import logging
import os
import socket
import struct
import sys
import time
import timeit
import traceback
import types
import warnings
from absl import flags
from absl._collections_abc import abc
from absl.logging import converter
import six
class ABSLLogger(logging.getLoggerClass()):
    """A logger that will create LogRecords while skipping some stack frames.

  This class maintains an internal list of filenames and method names
  for use when determining who called the currently executing stack
  frame.  Any method names from specific source files are skipped when
  walking backwards through the stack.

  Client code should use the register_frame_to_skip method to let the
  ABSLLogger know which method from which file should be
  excluded from the walk backwards through the stack.
  """
    _frames_to_skip = set()

    def findCaller(self, stack_info=False, stacklevel=1):
        """Finds the frame of the calling method on the stack.

    This method skips any frames registered with the
    ABSLLogger and any methods from this file, and whatever
    method is currently being used to generate the prefix for the log
    line.  Then it returns the file name, line number, and method name
    of the calling method.  An optional fourth item may be returned,
    callers who only need things from the first three are advised to
    always slice or index the result rather than using direct unpacking
    assignment.

    Args:
      stack_info: bool, when True, include the stack trace as a fourth item
          returned.  On Python 3 there are always four items returned - the
          fourth will be None when this is False.  On Python 2 the stdlib
          base class API only returns three items.  We do the same when this
          new parameter is unspecified or False for compatibility.

    Returns:
      (filename, lineno, methodname[, sinfo]) of the calling method.
    """
        f_to_skip = ABSLLogger._frames_to_skip
        frame = sys._getframe(2)
        while frame:
            code = frame.f_code
            if _LOGGING_FILE_PREFIX not in code.co_filename and (code.co_filename, code.co_name, code.co_firstlineno) not in f_to_skip and ((code.co_filename, code.co_name) not in f_to_skip):
                if six.PY2 and (not stack_info):
                    return (code.co_filename, frame.f_lineno, code.co_name)
                else:
                    sinfo = None
                    if stack_info:
                        out = io.StringIO()
                        out.write(u'Stack (most recent call last):\n')
                        traceback.print_stack(frame, file=out)
                        sinfo = out.getvalue().rstrip(u'\n')
                    return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
            frame = frame.f_back

    def critical(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'CRITICAL'."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'FATAL'."""
        self.log(logging.FATAL, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'ERROR'."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'WARN'."""
        if six.PY3:
            warnings.warn("The 'warn' method is deprecated, use 'warning' instead", DeprecationWarning, 2)
        self.log(logging.WARN, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'WARNING'."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'INFO'."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Logs 'msg % args' with severity 'DEBUG'."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """Logs a message at a cetain level substituting in the supplied arguments.

    This method behaves differently in python and c++ modes.

    Args:
      level: int, the standard logging level at which to log the message.
      msg: str, the text of the message to log.
      *args: The arguments to substitute in the message.
      **kwargs: The keyword arguments to substitute in the message.
    """
        if level >= logging.FATAL:
            extra = kwargs.setdefault('extra', {})
            extra[_ABSL_LOG_FATAL] = True
        super(ABSLLogger, self).log(level, msg, *args, **kwargs)

    def handle(self, record):
        """Calls handlers without checking Logger.disabled.

    Non-root loggers are set to disabled after setup with logging.config if
    it's not explicitly specified. Historically, absl logging will not be
    disabled by that. To maintaining this behavior, this function skips
    checking the Logger.disabled bit.

    This logger can still be disabled by adding a filter that filters out
    everything.

    Args:
      record: logging.LogRecord, the record to handle.
    """
        if self.filter(record):
            self.callHandlers(record)

    @classmethod
    def register_frame_to_skip(cls, file_name, function_name, line_number=None):
        """Registers a function name to skip when walking the stack.

    The ABSLLogger sometimes skips method calls on the stack
    to make the log messages meaningful in their appropriate context.
    This method registers a function from a particular file as one
    which should be skipped.

    Args:
      file_name: str, the name of the file that contains the function.
      function_name: str, the name of the function to skip.
      line_number: int, if provided, only the function with this starting line
          number will be skipped. Otherwise, all functions with the same name
          in the file will be skipped.
    """
        if line_number is not None:
            cls._frames_to_skip.add((file_name, function_name, line_number))
        else:
            cls._frames_to_skip.add((file_name, function_name))