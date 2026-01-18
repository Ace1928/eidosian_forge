import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_extract_drive_letter(self):
    extract = urlutils._win32_extract_drive_letter
    self.assertEqual(('file:///C:', '/foo'), extract('file://', '/C:/foo'))
    self.assertEqual(('file:///d|', '/path'), extract('file://', '/d|/path'))
    self.assertRaises(urlutils.InvalidURL, extract, 'file://', '/path')
    self.assertEqual(('file:///C:', '/'), extract('file://', '/C:/'))
    self.assertRaises(urlutils.InvalidURL, extract, 'file://', '/C:')
    self.assertRaises(urlutils.InvalidURL, extract, 'file://', '/C')
    self.assertRaises(urlutils.InvalidURL, extract, 'file://', '/C:ool')