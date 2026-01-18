import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_win32_local_path_to_url(self):
    to_url = urlutils._win32_local_path_to_url
    self.assertEqual('file:///C:/path/to/foo', to_url('C:/path/to/foo'))
    self.assertEqual('file:///C:/path/to/f%20oo', to_url('C:/path/to/f oo'))
    self.assertEqual('file:///', to_url('/'))
    self.assertEqual('file:///C:/path/to/foo%2Cbar', to_url('C:/path/to/foo,bar'))
    try:
        result = to_url('d:/path/to/räksmörgås')
    except UnicodeError:
        raise TestSkipped('local encoding cannot handle unicode')
    self.assertEqual('file:///D:/path/to/r%C3%A4ksm%C3%B6rg%C3%A5s', result)
    self.assertIsInstance(result, str)