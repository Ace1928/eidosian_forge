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
class TestShaFileByName(tests.TestCaseInTempDir):

    def test_sha_empty(self):
        self.build_tree_contents([('foo', b'')])
        expected_sha = osutils.sha_string(b'')
        self.assertEqual(expected_sha, osutils.sha_file_by_name('foo'))

    def test_sha_mixed_endings(self):
        text = b'test\r\nwith\nall\rpossible line endings\r\n'
        self.build_tree_contents([('foo', text)])
        expected_sha = osutils.sha_string(text)
        self.assertEqual(expected_sha, osutils.sha_file_by_name('foo'))