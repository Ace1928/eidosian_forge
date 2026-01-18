from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
def test_read_and_write_file_ordering(self):
    """Tests that rthru_file and wthru_file work when run together."""
    self._run_throughput_test('rthru_file,wthru_file', 1, 1)
    self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'fan')
    if not RUN_S3_TESTS:
        self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'slice')
        self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'both')