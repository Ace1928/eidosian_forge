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
def test_rename_exception(self):
    try:
        osutils.rename('nonexistent_path', 'different_nonexistent_path')
    except OSError as e:
        self.assertEqual(e.old_filename, 'nonexistent_path')
        self.assertEqual(e.new_filename, 'different_nonexistent_path')
        self.assertTrue('nonexistent_path' in e.strerror)
        self.assertTrue('different_nonexistent_path' in e.strerror)