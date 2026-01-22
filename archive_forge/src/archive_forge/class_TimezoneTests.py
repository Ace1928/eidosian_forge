import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
class TimezoneTests(PorcelainTestCase):

    def put_envs(self, value):
        self.overrideEnv('GIT_AUTHOR_DATE', value)
        self.overrideEnv('GIT_COMMITTER_DATE', value)

    def fallback(self, value):
        self.put_envs(value)
        self.assertRaises(porcelain.TimezoneFormatError, porcelain.get_user_timezones)

    def test_internal_format(self):
        self.put_envs('0 +0500')
        self.assertTupleEqual((18000, 18000), porcelain.get_user_timezones())

    def test_rfc_2822(self):
        self.put_envs('Mon, 20 Nov 1995 19:12:08 -0500')
        self.assertTupleEqual((-18000, -18000), porcelain.get_user_timezones())
        self.put_envs('Mon, 20 Nov 1995 19:12:08')
        self.assertTupleEqual((0, 0), porcelain.get_user_timezones())

    def test_iso8601(self):
        self.put_envs('1995-11-20T19:12:08-0501')
        self.assertTupleEqual((-18060, -18060), porcelain.get_user_timezones())
        self.put_envs('1995-11-20T19:12:08+0501')
        self.assertTupleEqual((18060, 18060), porcelain.get_user_timezones())
        self.put_envs('1995-11-20T19:12:08-05:01')
        self.assertTupleEqual((-18060, -18060), porcelain.get_user_timezones())
        self.put_envs('1995-11-20 19:12:08-05')
        self.assertTupleEqual((-18000, -18000), porcelain.get_user_timezones())
        self.put_envs('2006-07-03 17:18:44 +0200')
        self.assertTupleEqual((7200, 7200), porcelain.get_user_timezones())

    def test_missing_or_malformed(self):
        self.fallback('0 + 0500')
        self.fallback('a +0500')
        self.fallback('1995-11-20T19:12:08')
        self.fallback('1995-11-20T19:12:08-05:')
        self.fallback('1995.11.20')
        self.fallback('11/20/1995')
        self.fallback('20.11.1995')

    def test_different_envs(self):
        self.overrideEnv('GIT_AUTHOR_DATE', '0 +0500')
        self.overrideEnv('GIT_COMMITTER_DATE', '0 +0501')
        self.assertTupleEqual((18000, 18060), porcelain.get_user_timezones())

    def test_no_envs(self):
        local_timezone = time.localtime().tm_gmtoff
        self.put_envs('0 +0500')
        self.assertTupleEqual((18000, 18000), porcelain.get_user_timezones())
        self.overrideEnv('GIT_COMMITTER_DATE', None)
        self.assertTupleEqual((18000, local_timezone), porcelain.get_user_timezones())
        self.put_envs('0 +0500')
        self.overrideEnv('GIT_AUTHOR_DATE', None)
        self.assertTupleEqual((local_timezone, 18000), porcelain.get_user_timezones())
        self.put_envs('0 +0500')
        self.overrideEnv('GIT_AUTHOR_DATE', None)
        self.overrideEnv('GIT_COMMITTER_DATE', None)
        self.assertTupleEqual((local_timezone, local_timezone), porcelain.get_user_timezones())