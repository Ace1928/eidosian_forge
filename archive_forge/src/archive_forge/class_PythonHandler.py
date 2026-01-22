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
class PythonHandler(logging.StreamHandler):
    """The handler class used by Abseil Python logging implementation."""

    def __init__(self, stream=None, formatter=None):
        super(PythonHandler, self).__init__(stream)
        self.setFormatter(formatter or PythonFormatter())

    def start_logging_to_file(self, program_name=None, log_dir=None):
        """Starts logging messages to files instead of standard error."""
        FLAGS.logtostderr = False
        actual_log_dir, file_prefix, symlink_prefix = find_log_dir_and_names(program_name=program_name, log_dir=log_dir)
        basename = '%s.INFO.%s.%d' % (file_prefix, time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())), os.getpid())
        filename = os.path.join(actual_log_dir, basename)
        if six.PY2:
            self.stream = open(filename, 'a')
        else:
            self.stream = open(filename, 'a', encoding='utf-8')
        if getattr(os, 'symlink', None):
            symlink = os.path.join(actual_log_dir, symlink_prefix + '.INFO')
            try:
                if os.path.islink(symlink):
                    os.unlink(symlink)
                os.symlink(os.path.basename(filename), symlink)
            except EnvironmentError:
                pass

    def use_absl_log_file(self, program_name=None, log_dir=None):
        """Conditionally logs to files, based on --logtostderr."""
        if FLAGS['logtostderr'].value:
            self.stream = sys.stderr
        else:
            self.start_logging_to_file(program_name=program_name, log_dir=log_dir)

    def flush(self):
        """Flushes all log files."""
        self.acquire()
        try:
            self.stream.flush()
        except (EnvironmentError, ValueError):
            pass
        finally:
            self.release()

    def _log_to_stderr(self, record):
        """Emits the record to stderr.

    This temporarily sets the handler stream to stderr, calls
    StreamHandler.emit, then reverts the stream back.

    Args:
      record: logging.LogRecord, the record to log.
    """
        old_stream = self.stream
        self.stream = sys.stderr
        try:
            super(PythonHandler, self).emit(record)
        finally:
            self.stream = old_stream

    def emit(self, record):
        """Prints a record out to some streams.

    If FLAGS.logtostderr is set, it will print to sys.stderr ONLY.
    If FLAGS.alsologtostderr is set, it will print to sys.stderr.
    If FLAGS.logtostderr is not set, it will log to the stream
      associated with the current thread.

    Args:
      record: logging.LogRecord, the record to emit.
    """
        level = record.levelno
        if not FLAGS.is_parsed():
            global _warn_preinit_stderr
            if _warn_preinit_stderr:
                sys.stderr.write('WARNING: Logging before flag parsing goes to stderr.\n')
                _warn_preinit_stderr = False
            self._log_to_stderr(record)
        elif FLAGS['logtostderr'].value:
            self._log_to_stderr(record)
        else:
            super(PythonHandler, self).emit(record)
            stderr_threshold = converter.string_to_standard(FLAGS['stderrthreshold'].value)
            if (FLAGS['alsologtostderr'].value or level >= stderr_threshold) and self.stream != sys.stderr:
                self._log_to_stderr(record)
        if _is_absl_fatal_record(record):
            self.flush()
            os.abort()

    def close(self):
        """Closes the stream to which we are writing."""
        self.acquire()
        try:
            self.flush()
            try:
                user_managed = (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__)
                if self.stream not in user_managed and (not hasattr(self.stream, 'isatty') or not self.stream.isatty()):
                    self.stream.close()
            except ValueError:
                pass
            super(PythonHandler, self).close()
        finally:
            self.release()