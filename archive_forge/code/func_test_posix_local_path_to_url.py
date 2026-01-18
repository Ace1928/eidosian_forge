import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_posix_local_path_to_url(self):
    to_url = urlutils._posix_local_path_to_url
    self.assertEqual('file:///path/to/foo', to_url('/path/to/foo'))
    self.assertEqual('file:///path/to/foo%2Cbar', to_url('/path/to/foo,bar'))
    try:
        result = to_url('/path/to/räksmörgås')
    except UnicodeError:
        raise TestSkipped('local encoding cannot handle unicode')
    self.assertEqual('file:///path/to/r%C3%A4ksm%C3%B6rg%C3%A5s', result)
    self.assertTrue(isinstance(result, str))