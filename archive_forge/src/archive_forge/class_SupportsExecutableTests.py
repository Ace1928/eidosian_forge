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
class SupportsExecutableTests(tests.TestCaseInTempDir):

    def test_returns_bool(self):
        self.assertIsInstance(osutils.supports_executable(self.test_dir), bool)