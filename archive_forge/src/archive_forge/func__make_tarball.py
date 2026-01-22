import unittest
import os
import sys
import tarfile
from os.path import splitdrive
import warnings
from distutils import archive_util
from distutils.archive_util import (check_archive_formats, make_tarball,
from distutils.spawn import find_executable, spawn
from distutils.tests import support
from test.support import patch
from test.support.os_helper import change_cwd
from test.support.warnings_helper import check_warnings
def _make_tarball(self, tmpdir, target_name, suffix, **kwargs):
    tmpdir2 = self.mkdtemp()
    unittest.skipUnless(splitdrive(tmpdir)[0] == splitdrive(tmpdir2)[0], 'source and target should be on same drive')
    base_name = os.path.join(tmpdir2, target_name)
    with change_cwd(tmpdir):
        make_tarball(splitdrive(base_name)[1], 'dist', **kwargs)
    tarball = base_name + suffix
    self.assertTrue(os.path.exists(tarball))
    self.assertEqual(self._tarinfo(tarball), self._created_files)