import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
class IgnoreFilterTests(TestCase):

    def test_included(self):
        filter = IgnoreFilter([b'a.c', b'b.c'])
        self.assertTrue(filter.is_ignored(b'a.c'))
        self.assertIs(None, filter.is_ignored(b'c.c'))
        self.assertEqual([Pattern(b'a.c')], list(filter.find_matching(b'a.c')))
        self.assertEqual([], list(filter.find_matching(b'c.c')))

    def test_included_ignorecase(self):
        filter = IgnoreFilter([b'a.c', b'b.c'], ignorecase=False)
        self.assertTrue(filter.is_ignored(b'a.c'))
        self.assertFalse(filter.is_ignored(b'A.c'))
        filter = IgnoreFilter([b'a.c', b'b.c'], ignorecase=True)
        self.assertTrue(filter.is_ignored(b'a.c'))
        self.assertTrue(filter.is_ignored(b'A.c'))
        self.assertTrue(filter.is_ignored(b'A.C'))

    def test_excluded(self):
        filter = IgnoreFilter([b'a.c', b'b.c', b'!c.c'])
        self.assertFalse(filter.is_ignored(b'c.c'))
        self.assertIs(None, filter.is_ignored(b'd.c'))
        self.assertEqual([Pattern(b'!c.c')], list(filter.find_matching(b'c.c')))
        self.assertEqual([], list(filter.find_matching(b'd.c')))

    def test_include_exclude_include(self):
        filter = IgnoreFilter([b'a.c', b'!a.c', b'a.c'])
        self.assertTrue(filter.is_ignored(b'a.c'))
        self.assertEqual([Pattern(b'a.c'), Pattern(b'!a.c'), Pattern(b'a.c')], list(filter.find_matching(b'a.c')))

    def test_manpage(self):
        filter = IgnoreFilter([b'/*', b'!/foo', b'/foo/*', b'!/foo/bar'])
        self.assertTrue(filter.is_ignored(b'a.c'))
        self.assertTrue(filter.is_ignored(b'foo/blie'))
        self.assertFalse(filter.is_ignored(b'foo'))
        self.assertFalse(filter.is_ignored(b'foo/bar'))
        self.assertFalse(filter.is_ignored(b'foo/bar/'))
        self.assertFalse(filter.is_ignored(b'foo/bar/bloe'))

    def test_regex_special(self):
        filter = IgnoreFilter([b'/foo\\[bar\\]', b'/foo'])
        self.assertTrue(filter.is_ignored('foo'))
        self.assertTrue(filter.is_ignored('foo[bar]'))