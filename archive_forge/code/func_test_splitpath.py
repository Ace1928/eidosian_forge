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
def test_splitpath(self):

    def check(expected, path):
        self.assertEqual(expected, osutils.splitpath(path))
    check(['a'], 'a')
    check(['a', 'b'], 'a/b')
    check(['a', 'b'], 'a/./b')
    check(['a', '.b'], 'a/.b')
    if os.path.sep == '\\':
        check(['a', '.b'], 'a\\.b')
    else:
        check(['a\\.b'], 'a\\.b')
    self.assertRaises(errors.BzrError, osutils.splitpath, 'a/../b')