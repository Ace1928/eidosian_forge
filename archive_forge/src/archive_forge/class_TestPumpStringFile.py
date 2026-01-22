import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestPumpStringFile(tests.TestCase):

    def test_empty(self):
        output = BytesIO()
        osutils.pump_string_file(b'', output)
        self.assertEqual(b'', output.getvalue())

    def test_more_than_segment_size(self):
        output = BytesIO()
        osutils.pump_string_file(b'123456789', output, 2)
        self.assertEqual(b'123456789', output.getvalue())

    def test_segment_size(self):
        output = BytesIO()
        osutils.pump_string_file(b'12', output, 2)
        self.assertEqual(b'12', output.getvalue())

    def test_segment_size_multiple(self):
        output = BytesIO()
        osutils.pump_string_file(b'1234', output, 2)
        self.assertEqual(b'1234', output.getvalue())