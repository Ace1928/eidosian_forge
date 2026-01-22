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
class TestParentDirectories(tests.TestCaseInTempDir):
    """Test osutils.parent_directories()"""

    def test_parent_directories(self):
        self.assertEqual([], osutils.parent_directories('a'))
        self.assertEqual(['a'], osutils.parent_directories('a/b'))
        self.assertEqual(['a/b', 'a'], osutils.parent_directories('a/b/c'))