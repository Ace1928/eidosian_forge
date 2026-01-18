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
def test_rename_missing_file(self):
    with open('a', 'wb') as a:
        a.write(b'foo\n')
    try:
        osutils._win32_rename('b', 'a')
    except OSError as e:
        self.assertEqual(errno.ENOENT, e.errno)
    self.assertFileEqual(b'foo\n', 'a')