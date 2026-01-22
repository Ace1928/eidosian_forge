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
class TestTestLoader(tests.TestCase):
    """Tests for the test loader."""

    def _get_loader_and_module(self):
        """Gets a TestLoader and a module with one test in it."""
        loader = TestUtil.TestLoader()
        module = {}

        class Stub(tests.TestCase):

            def test_foo(self):
                pass

        class MyModule:
            pass
        MyModule.a_class = Stub
        module = MyModule()
        module.__name__ = 'fake_module'
        return (loader, module)

    def test_module_no_load_tests_attribute_loads_classes(self):
        loader, module = self._get_loader_and_module()
        self.assertEqual(1, loader.loadTestsFromModule(module).countTestCases())

    def test_module_load_tests_attribute_gets_called(self):
        loader, module = self._get_loader_and_module()

        def load_tests(loader, standard_tests, pattern):
            result = loader.suiteClass()
            for test in tests.iter_suite_tests(standard_tests):
                result.addTests([test, test])
            return result
        module.__class__.load_tests = staticmethod(load_tests)
        self.assertEqual(2 * [str(module.a_class('test_foo'))], list(map(str, loader.loadTestsFromModule(module))))

    def test_load_tests_from_module_name_smoke_test(self):
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
        self.assertEqual(['breezy.tests.test_sampler.DemoTest.test_nothing'], _test_ids(suite))

    def test_load_tests_from_module_name_with_bogus_module_name(self):
        loader = TestUtil.TestLoader()
        self.assertRaises(ImportError, loader.loadTestsFromModuleName, 'bogus')