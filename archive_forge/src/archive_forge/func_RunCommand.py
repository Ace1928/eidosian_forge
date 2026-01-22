from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import sys
import tempfile
import six
import boto
from boto.utils import get_utf8able_str
from gslib import project_id
from gslib import wildcard_iterator
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command_runner import CommandRunner
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.constants import UTF8
from gslib.utils.text_util import print_to_fd
def RunCommand(self, command_name, args=None, headers=None, debug=0, return_stdout=False, return_stderr=False, return_log_handler=False, cwd=None):
    """Method for calling gslib.command_runner.CommandRunner.

    Passes parallel_operations=False for all tests, optionally saving/returning
    stdout output. We run all tests multi-threaded, to exercise those more
    complicated code paths.
    TODO: Change to run with parallel_operations=True for all tests. At
    present when you do this it causes many test failures.

    Args:
      command_name: The name of the command being run.
      args: Command-line args (arg0 = actual arg, not command name ala bash).
      headers: Dictionary containing optional HTTP headers to pass to boto.
      debug: Debug level to pass in to boto connection (range 0..3).
      return_stdout: If True, will save and return stdout produced by command.
      return_stderr: If True, will save and return stderr produced by command.
      return_log_handler: If True, will return a MockLoggingHandler instance
           that was attached to the command's logger while running.
      cwd: The working directory that should be switched to before running the
           command. The working directory will be reset back to its original
           value after running the command. If not specified, the working
           directory is left unchanged.

    Returns:
      One or a tuple of requested return values, depending on whether
      return_stdout, return_stderr, and/or return_log_handler were specified.
      Return Types:
        stdout - str (binary in Py2, text in Py3)
        stderr - str (binary in Py2, text in Py3)
        log_handler - MockLoggingHandler
    """
    args = args or []
    command_line = six.ensure_text(' '.join([command_name] + args))
    if self.is_debugging:
        print_to_fd('\nRunCommand of {}\n'.format(command_line), file=self.stderr_save)
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    stdout = sys.stdout.read()
    stderr = sys.stderr.read()
    if stdout:
        self.accumulated_stdout.append(stdout)
    if stderr:
        self.accumulated_stderr.append(stderr)
    sys.stdout.seek(0)
    sys.stderr.seek(0)
    sys.stdout.truncate()
    sys.stderr.truncate()
    mock_log_handler = MockLoggingHandler()
    logging.getLogger(command_name).addHandler(mock_log_handler)
    if debug:
        logging.getLogger(command_name).setLevel(logging.DEBUG)
    try:
        with WorkingDirectory(cwd):
            self.command_runner.RunNamedCommand(command_name, args=args, headers=headers, debug=debug, parallel_operations=False, do_shutdown=False)
    finally:
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        if six.PY2:
            stdout = sys.stdout.read()
            stderr = sys.stderr.read()
        else:
            try:
                stdout = sys.stdout.read()
                stderr = sys.stderr.read()
            except UnicodeDecodeError:
                sys.stdout.seek(0)
                sys.stderr.seek(0)
                stdout = sys.stdout.buffer.read()
                stderr = sys.stderr.buffer.read()
        logging.getLogger(command_name).removeHandler(mock_log_handler)
        mock_log_handler.close()
        log_output = '\n'.join(('%s:\n  ' % level + '\n  '.join(records) for level, records in six.iteritems(mock_log_handler.messages) if records))
        _id = six.ensure_text(self.id())
        if self.is_debugging and log_output:
            print_to_fd('==== logging RunCommand {} {} ====\n'.format(_id, command_line), file=self.stderr_save)
            print_to_fd(log_output, file=self.stderr_save)
            print_to_fd('\n==== end logging ====\n', file=self.stderr_save)
        if self.is_debugging and stdout:
            print_to_fd('==== stdout RunCommand {} {} ====\n'.format(_id, command_line), file=self.stderr_save)
            print_to_fd(stdout, file=self.stderr_save)
            print_to_fd('==== end stdout ====\n', file=self.stderr_save)
        if self.is_debugging and stderr:
            print_to_fd('==== stderr RunCommand {} {} ====\n'.format(_id, command_line), file=self.stderr_save)
            print_to_fd(stderr, file=self.stderr_save)
            print_to_fd('==== end stderr ====\n', file=self.stderr_save)
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        sys.stdout.truncate()
        sys.stderr.truncate()
    to_return = []
    if return_stdout:
        to_return.append(stdout)
    if return_stderr:
        to_return.append(stderr)
    if return_log_handler:
        to_return.append(mock_log_handler)
    if len(to_return) == 1:
        return to_return[0]
    return tuple(to_return)