import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_sibling_posix(self):
    self._with_posix_paths()
    self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b', 'file:///a/c')
    self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b/', 'file:///a/c')
    self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b/', 'file:///a/c/')