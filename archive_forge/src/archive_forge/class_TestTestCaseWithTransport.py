import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestTestCaseWithTransport(tests.TestCaseWithTransport):
    """Tests for the convenience functions TestCaseWithTransport introduces."""

    def test_get_readonly_url_none(self):
        from ..transport.readonly import ReadonlyTransportDecorator
        self.vfs_transport_factory = memory.MemoryServer
        self.transport_readonly_server = None
        url = self.get_readonly_url()
        url2 = self.get_readonly_url('foo/bar')
        t = transport.get_transport_from_url(url)
        t2 = transport.get_transport_from_url(url2)
        self.assertIsInstance(t, ReadonlyTransportDecorator)
        self.assertIsInstance(t2, ReadonlyTransportDecorator)
        self.assertEqual(t2.base[:-1], t.abspath('foo/bar'))

    def test_get_readonly_url_http(self):
        from ..transport.http.urllib import HttpTransport
        from .http_server import HttpServer
        self.transport_server = test_server.LocalURLServer
        self.transport_readonly_server = HttpServer
        url = self.get_readonly_url()
        url2 = self.get_readonly_url('foo/bar')
        t = transport.get_transport_from_url(url)
        t2 = transport.get_transport_from_url(url2)
        self.assertIsInstance(t, HttpTransport)
        self.assertIsInstance(t2, HttpTransport)
        self.assertEqual(t2.base[:-1], t.abspath('foo/bar'))

    def test_is_directory(self):
        """Test assertIsDirectory assertion"""
        t = self.get_transport()
        self.build_tree(['a_dir/', 'a_file'], transport=t)
        self.assertIsDirectory('a_dir', t)
        self.assertRaises(AssertionError, self.assertIsDirectory, 'a_file', t)
        self.assertRaises(AssertionError, self.assertIsDirectory, 'not_here', t)

    def test_make_branch_builder(self):
        builder = self.make_branch_builder('dir')
        rev_id = builder.build_commit()
        self.assertPathExists('dir')
        a_dir = controldir.ControlDir.open('dir')
        self.assertRaises(errors.NoWorkingTree, a_dir.open_workingtree)
        a_branch = a_dir.open_branch()
        builder_branch = builder.get_branch()
        self.assertEqual(a_branch.base, builder_branch.base)
        self.assertEqual((1, rev_id), builder_branch.last_revision_info())
        self.assertEqual((1, rev_id), a_branch.last_revision_info())