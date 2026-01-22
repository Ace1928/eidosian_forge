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
def _test_and_check_logging_communicate_errors(self, log_errors=None, attempts=None):
    mock = self.useFixture(fixtures.MockPatch('subprocess.Popen.communicate', side_effect=OSError(errno.EAGAIN, 'fake-test')))
    fixture = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
    kwargs = {}
    if log_errors:
        kwargs.update({'log_errors': log_errors})
    if attempts:
        kwargs.update({'attempts': attempts})
    self.assertRaises(OSError, processutils.execute, '/usr/bin/env', 'false', **kwargs)
    self.assertEqual(attempts if attempts else 1, mock.mock.call_count)
    self.assertIn('Got an OSError', fixture.output)
    self.assertIn('errno: %d' % errno.EAGAIN, fixture.output)
    self.assertIn("'/usr/bin/env false'", fixture.output)