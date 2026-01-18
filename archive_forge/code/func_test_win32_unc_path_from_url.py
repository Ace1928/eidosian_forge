import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_unc_path_from_url(self):
    from_url = urlutils._win32_local_path_from_url
    self.assertEqual('//HOST/path', from_url('file://HOST/path'))
    self.assertEqual('//HOST/path', from_url('file://HOST/path,branch=foo'))
    self.assertRaises(urlutils.InvalidURL, from_url, 'file:////HOST/path')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file://///HOST/path')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file://////HOST/path')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file://C:/path')