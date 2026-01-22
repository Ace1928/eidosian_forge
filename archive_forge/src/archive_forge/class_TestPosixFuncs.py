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
class TestPosixFuncs(tests.TestCase):
    """Test that the posix version of normpath returns an appropriate path
       when used with 2 leading slashes."""

    def test_normpath(self):
        self.assertEqual('/etc/shadow', osutils._posix_normpath('/etc/shadow'))
        self.assertEqual('/etc/shadow', osutils._posix_normpath('//etc/shadow'))
        self.assertEqual('/etc/shadow', osutils._posix_normpath('///etc/shadow'))