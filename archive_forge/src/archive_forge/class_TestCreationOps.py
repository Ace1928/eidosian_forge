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
class TestCreationOps(tests.TestCaseInTempDir):
    _test_needs_features = [features.chown_feature]

    def setUp(self):
        super().setUp()
        self.overrideAttr(os, 'chown', self._dummy_chown)
        self.path = self.uid = self.gid = None

    def _dummy_chown(self, path, uid, gid):
        self.path, self.uid, self.gid = (path, uid, gid)

    def test_copy_ownership_from_path(self):
        """copy_ownership_from_path test with specified src."""
        ownsrc = '/'
        open('test_file', 'w').close()
        osutils.copy_ownership_from_path('test_file', ownsrc)
        s = os.stat(ownsrc)
        self.assertEqual(self.path, 'test_file')
        self.assertEqual(self.uid, s.st_uid)
        self.assertEqual(self.gid, s.st_gid)

    def test_copy_ownership_nonesrc(self):
        """copy_ownership_from_path test with src=None."""
        open('test_file', 'w').close()
        osutils.copy_ownership_from_path('test_file')
        s = os.stat('..')
        self.assertEqual(self.path, 'test_file')
        self.assertEqual(self.uid, s.st_uid)
        self.assertEqual(self.gid, s.st_gid)