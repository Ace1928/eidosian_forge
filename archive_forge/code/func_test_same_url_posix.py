import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_same_url_posix(self):
    self._with_posix_paths()
    self.assertEqual('', urlutils.file_relpath('file:///a', 'file:///a'))
    self.assertEqual('', urlutils.file_relpath('file:///a', 'file:///a/'))
    self.assertEqual('', urlutils.file_relpath('file:///a/', 'file:///a'))