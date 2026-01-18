import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_same_url_win32(self):
    self._with_win32_paths()
    self.assertEqual('', urlutils.file_relpath('file:///A:/', 'file:///A:/'))
    self.assertEqual('', urlutils.file_relpath('file:///A|/', 'file:///A:/'))
    self.assertEqual('', urlutils.file_relpath('file:///A:/b/', 'file:///A:/b/'))
    self.assertEqual('', urlutils.file_relpath('file:///A:/b', 'file:///A:/b/'))
    self.assertEqual('', urlutils.file_relpath('file:///A:/b/', 'file:///A:/b'))