import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
class SampleLoadTestsPackage(fixtures.Fixture):
    """Creates a test suite package using load_tests."""

    def __init__(self):
        super().__init__()
        self.package = fixtures.PythonPackage('discoverexample', [('__init__.py', _b('\nfrom testtools import TestCase, clone_test_with_new_id\n\nclass TestExample(TestCase):\n    def test_foo(self):\n        pass\n\ndef load_tests(loader, tests, pattern):\n    tests.addTest(clone_test_with_new_id(tests._tests[1]._tests[0], "fred"))\n    return tests\n'))])

    def setUp(self):
        super().setUp()
        self.useFixture(self.package)
        self.addCleanup(sys.path.remove, self.package.base)