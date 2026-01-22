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
class GetFsTypeTests(tests.TestCaseInTempDir):

    def test_returns_string_or_none(self):
        ret = osutils.get_fs_type(self.test_dir)
        self.assertTrue(isinstance(ret, str) or ret is None)

    def test_returns_most_specific(self):
        self.overrideAttr(osutils, '_FILESYSTEM_FINDER', osutils.MtabFilesystemFinder([(b'/', 'ext4'), (b'/home', 'vfat'), (b'/home/jelmer', 'ext2')]))
        self.assertEqual(osutils.get_fs_type(b'/home/jelmer/blah'), 'ext2')
        self.assertEqual(osutils.get_fs_type('/home/jelmer/blah'), 'ext2')
        self.assertEqual(osutils.get_fs_type(b'/home/jelmer'), 'ext2')
        self.assertEqual(osutils.get_fs_type(b'/home/martin'), 'vfat')
        self.assertEqual(osutils.get_fs_type(b'/home'), 'vfat')
        self.assertEqual(osutils.get_fs_type(b'/other'), 'ext4')

    def test_returns_none(self):
        self.overrideAttr(osutils, '_FILESYSTEM_FINDER', osutils.MtabFilesystemFinder([]))
        self.assertIs(osutils.get_fs_type('/home/jelmer/blah'), None)
        self.assertIs(osutils.get_fs_type(b'/home/jelmer/blah'), None)
        self.assertIs(osutils.get_fs_type('/home/jelmer'), None)