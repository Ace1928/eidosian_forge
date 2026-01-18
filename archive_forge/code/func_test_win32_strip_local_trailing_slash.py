import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_strip_local_trailing_slash(self):
    strip = urlutils._win32_strip_local_trailing_slash
    self.assertEqual('file://', strip('file://'))
    self.assertEqual('file:///', strip('file:///'))
    self.assertEqual('file:///C', strip('file:///C'))
    self.assertEqual('file:///C:', strip('file:///C:'))
    self.assertEqual('file:///d|', strip('file:///d|'))
    self.assertEqual('file:///C:/', strip('file:///C:/'))
    self.assertEqual('file:///C:/a', strip('file:///C:/a/'))