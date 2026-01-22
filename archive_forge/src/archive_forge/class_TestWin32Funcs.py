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
class TestWin32Funcs(tests.TestCase):
    """Test that _win32 versions of os utilities return appropriate paths."""

    def test_abspath(self):
        self.requireFeature(features.win32_feature)
        self.assertEqual('C:/foo', osutils._win32_abspath('C:\\foo'))
        self.assertEqual('C:/foo', osutils._win32_abspath('C:/foo'))
        self.assertEqual('//HOST/path', osutils._win32_abspath('\\\\HOST\\path'))
        self.assertEqual('//HOST/path', osutils._win32_abspath('//HOST/path'))

    def test_realpath(self):
        self.assertEqual('C:/foo', osutils._win32_realpath('C:\\foo'))
        self.assertEqual('C:/foo', osutils._win32_realpath('C:/foo'))

    def test_pathjoin(self):
        self.assertEqual('path/to/foo', osutils._win32_pathjoin('path', 'to', 'foo'))
        self.assertEqual('C:/foo', osutils._win32_pathjoin('path\\to', 'C:\\foo'))
        self.assertEqual('C:/foo', osutils._win32_pathjoin('path/to', 'C:/foo'))
        self.assertEqual('path/to/foo', osutils._win32_pathjoin('path/to/', 'foo'))

    def test_pathjoin_late_bugfix(self):
        expected = 'C:/foo'
        self.assertEqual(expected, osutils._win32_pathjoin('C:/path/to/', '/foo'))
        self.assertEqual(expected, osutils._win32_pathjoin('C:\\path\\to\\', '\\foo'))

    def test_normpath(self):
        self.assertEqual('path/to/foo', osutils._win32_normpath('path\\\\from\\..\\to\\.\\foo'))
        self.assertEqual('path/to/foo', osutils._win32_normpath('path//from/../to/./foo'))

    def test_getcwd(self):
        cwd = osutils._win32_getcwd()
        os_cwd = os.getcwd()
        self.assertEqual(os_cwd[1:].replace('\\', '/'), cwd[1:])
        self.assertEqual(os_cwd[0].upper(), cwd[0])

    def test_fixdrive(self):
        self.assertEqual('H:/foo', osutils._win32_fixdrive('h:/foo'))
        self.assertEqual('H:/foo', osutils._win32_fixdrive('H:/foo'))
        self.assertEqual('C:\\foo', osutils._win32_fixdrive('c:\\foo'))