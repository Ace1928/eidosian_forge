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
class TestFilteredByModuleTestLoader(tests.TestCase):

    def _create_loader(self, test_list):
        id_filter = tests.TestIdList(test_list)
        loader = TestUtil.FilteredByModuleTestLoader(id_filter.refers_to)
        return loader

    def test_load_tests(self):
        test_list = ['breezy.tests.test_sampler.DemoTest.test_nothing']
        loader = self._create_loader(test_list)
        suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
        self.assertEqual(test_list, _test_ids(suite))

    def test_exclude_tests(self):
        test_list = ['bogus']
        loader = self._create_loader(test_list)
        suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
        self.assertEqual([], _test_ids(suite))