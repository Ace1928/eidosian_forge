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
class TestUmask(tests.TestCaseInTempDir):

    def test_get_umask(self):
        if sys.platform == 'win32':
            self.assertEqual(0, osutils.get_umask())
            return
        orig_umask = osutils.get_umask()
        self.addCleanup(os.umask, orig_umask)
        os.umask(146)
        self.assertEqual(146, osutils.get_umask())
        os.umask(18)
        self.assertEqual(18, osutils.get_umask())
        os.umask(2)
        self.assertEqual(2, osutils.get_umask())
        os.umask(23)
        self.assertEqual(23, osutils.get_umask())