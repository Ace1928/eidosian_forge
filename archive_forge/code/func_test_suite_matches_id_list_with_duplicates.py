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
def test_suite_matches_id_list_with_duplicates(self):
    loader = TestUtil.TestLoader()
    suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
    dupes = loader.suiteClass()
    for test in tests.iter_suite_tests(suite):
        dupes.addTest(test)
        dupes.addTest(test)
    test_list = ['breezy.tests.test_sampler.DemoTest.test_nothing']
    not_found, duplicates = tests.suite_matches_id_list(dupes, test_list)
    self.assertEqual([], not_found)
    self.assertEqual(['breezy.tests.test_sampler.DemoTest.test_nothing'], duplicates)