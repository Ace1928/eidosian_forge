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
class TestLinks(tests.TestCaseInTempDir):

    def test_dereference_path(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        cwd = osutils.realpath('.')
        os.mkdir('bar')
        bar_path = osutils.pathjoin(cwd, 'bar')
        self.assertEqual(bar_path, osutils.realpath('./bar'))
        os.symlink('bar', 'foo')
        self.assertEqual(bar_path, osutils.realpath('./foo'))
        foo_path = osutils.pathjoin(cwd, 'foo')
        self.assertEqual(foo_path, osutils.dereference_path('./foo'))
        os.mkdir('bar/baz')
        baz_path = osutils.pathjoin(bar_path, 'baz')
        self.assertEqual(baz_path, osutils.dereference_path('./foo/baz'))
        self.assertEqual(baz_path, osutils.dereference_path('foo/baz'))
        foo_baz_path = osutils.pathjoin(foo_path, 'baz')
        self.assertEqual(baz_path, osutils.dereference_path(foo_baz_path))

    def test_changing_access(self):
        with open('file', 'w') as f:
            f.write('monkey')
        osutils.make_readonly('file')
        mode = os.lstat('file').st_mode
        self.assertEqual(mode, mode & 261997)
        osutils.make_writable('file')
        mode = os.lstat('file').st_mode
        self.assertEqual(mode, mode | 128)
        if osutils.supports_symlinks(self.test_dir):
            os.symlink('nonexistent', 'dangling')
            osutils.make_readonly('dangling')
            osutils.make_writable('dangling')