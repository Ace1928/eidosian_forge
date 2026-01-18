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
def test_segment_size_multiple(self):
    output = BytesIO()
    osutils.pump_string_file(b'1234', output, 2)
    self.assertEqual(b'1234', output.getvalue())