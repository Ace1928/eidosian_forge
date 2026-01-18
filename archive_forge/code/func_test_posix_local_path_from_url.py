import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_posix_local_path_from_url(self):
    from_url = urlutils._posix_local_path_from_url
    self.assertEqual('/path/to/foo', from_url('file:///path/to/foo'))
    self.assertEqual('/path/to/foo', from_url('file:///path/to/foo,branch=foo'))
    self.assertEqual('/path/to/räksmörgås', from_url('file:///path/to/r%C3%A4ksm%C3%B6rg%C3%A5s'))
    self.assertEqual('/path/to/räksmörgås', from_url('file:///path/to/r%c3%a4ksm%c3%b6rg%c3%a5s'))
    self.assertEqual('/path/to/räksmörgås', from_url('file://localhost/path/to/r%c3%a4ksm%c3%b6rg%c3%a5s'))
    self.assertRaises(urlutils.InvalidURL, from_url, '/path/to/foo')
    self.assertRaises(urlutils.InvalidURL, from_url, 'file://remotehost/path/to/r%c3%a4ksm%c3%b6rg%c3%a5s')