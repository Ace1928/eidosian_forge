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
class TestLstat(tests.TestCaseInTempDir):

    def test_lstat_matches_fstat(self):
        if sys.platform == 'win32':
            self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
        with open('test-file.txt', 'wb') as f:
            f.write(b'some content\n')
            f.flush()
            self.assertEqualStat(osutils.fstat(f.fileno()), osutils.lstat('test-file.txt'))