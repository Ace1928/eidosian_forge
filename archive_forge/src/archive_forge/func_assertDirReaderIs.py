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
def assertDirReaderIs(self, expected, top, fs_enc=None):
    """Assert the right implementation for _walkdirs_utf8 is chosen."""
    osutils._selected_dir_reader = None
    self.assertEqual([((b'', top), [])], list(osutils._walkdirs_utf8('.', fs_enc=fs_enc)))
    self.assertIsInstance(osutils._selected_dir_reader, expected)