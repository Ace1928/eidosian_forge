import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
class ProcessExecutionErrorLoggingTest(test_base.BaseTestCase):

    def setUp(self):
        super(ProcessExecutionErrorLoggingTest, self).setUp()
        self.tmpfilename = self.create_tempfiles([['process_execution_error_logging_test', PROCESS_EXECUTION_ERROR_LOGGING_TEST]], ext='bash')[0]
        os.chmod(self.tmpfilename, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    def _test_and_check(self, log_errors=None, attempts=None):
        fixture = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        kwargs = {}
        if log_errors:
            kwargs.update({'log_errors': log_errors})
        if attempts:
            kwargs.update({'attempts': attempts})
        err = self.assertRaises(processutils.ProcessExecutionError, processutils.execute, self.tmpfilename, **kwargs)
        self.assertEqual(41, err.exit_code)
        self.assertIn(self.tmpfilename, fixture.output)

    def test_with_invalid_log_errors(self):
        self.assertRaises(processutils.InvalidArgumentError, processutils.execute, self.tmpfilename, log_errors='invalid')

    def test_with_log_errors_NONE(self):
        self._test_and_check(log_errors=None, attempts=None)

    def test_with_log_errors_final(self):
        self._test_and_check(log_errors=processutils.LOG_FINAL_ERROR, attempts=None)

    def test_with_log_errors_all(self):
        self._test_and_check(log_errors=processutils.LOG_ALL_ERRORS, attempts=None)

    def test_multiattempt_with_log_errors_NONE(self):
        self._test_and_check(log_errors=None, attempts=3)

    def test_multiattempt_with_log_errors_final(self):
        self._test_and_check(log_errors=processutils.LOG_FINAL_ERROR, attempts=3)

    def test_multiattempt_with_log_errors_all(self):
        self._test_and_check(log_errors=processutils.LOG_ALL_ERRORS, attempts=3)