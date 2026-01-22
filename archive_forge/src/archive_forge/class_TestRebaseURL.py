import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestRebaseURL(TestCase):
    """Test the behavior of rebase_url."""

    def test_non_relative(self):
        result = urlutils.rebase_url('file://foo', 'file://foo', 'file://foo/bar')
        self.assertEqual('file://foo', result)
        result = urlutils.rebase_url('/foo', 'file://foo', 'file://foo/bar')
        self.assertEqual('/foo', result)

    def test_different_ports(self):
        e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar:80', 'http://bar:81')
        self.assertEqual(str(e), "URLs differ by more than path: 'http://bar:80' and 'http://bar:81'")

    def test_different_hosts(self):
        e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar', 'http://baz')
        self.assertEqual(str(e), "URLs differ by more than path: 'http://bar' and 'http://baz'")

    def test_different_protocol(self):
        e = self.assertRaises(urlutils.InvalidRebaseURLs, urlutils.rebase_url, 'foo', 'http://bar', 'ftp://bar')
        self.assertEqual(str(e), "URLs differ by more than path: 'http://bar' and 'ftp://bar'")

    def test_rebase_success(self):
        self.assertEqual('../bar', urlutils.rebase_url('bar', 'http://baz/', 'http://baz/qux'))
        self.assertEqual('qux/bar', urlutils.rebase_url('bar', 'http://baz/qux', 'http://baz/'))
        self.assertEqual('.', urlutils.rebase_url('foo', 'http://bar/', 'http://bar/foo/'))
        self.assertEqual('qux/bar', urlutils.rebase_url('../bar', 'http://baz/qux/foo', 'http://baz/'))

    def test_determine_relative_path(self):
        self.assertEqual('../../baz/bar', urlutils.determine_relative_path('/qux/quxx', '/baz/bar'))
        self.assertEqual('..', urlutils.determine_relative_path('/bar/baz', '/bar'))
        self.assertEqual('baz', urlutils.determine_relative_path('/bar', '/bar/baz'))
        self.assertEqual('.', urlutils.determine_relative_path('/bar', '/bar'))