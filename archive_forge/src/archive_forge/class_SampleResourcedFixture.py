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
class SampleResourcedFixture(fixtures.Fixture):
    """Creates a test suite that uses testresources."""

    def __init__(self):
        super().__init__()
        self.package = fixtures.PythonPackage('resourceexample', [('__init__.py', _b("\nfrom fixtures import Fixture\nfrom testresources import (\n    FixtureResource,\n    OptimisingTestSuite,\n    ResourcedTestCase,\n    )\nfrom testtools import TestCase\n\nclass Printer(Fixture):\n\n    def setUp(self):\n        super(Printer, self).setUp()\n        print('Setting up Printer')\n\n    def reset(self):\n        pass\n\nclass TestFoo(TestCase, ResourcedTestCase):\n    # When run, this will print just one Setting up Printer, unless the\n    # OptimisingTestSuite is not honoured, when one per test case will print.\n    resources=[('res', FixtureResource(Printer()))]\n    def test_bar(self):\n        pass\n    def test_foo(self):\n        pass\n    def test_quux(self):\n        pass\ndef test_suite():\n    from unittest import TestLoader\n    return OptimisingTestSuite(TestLoader().loadTestsFromName(__name__))\n"))])

    def setUp(self):
        super().setUp()
        self.useFixture(self.package)
        self.addCleanup(testtools.__path__.remove, self.package.base)
        testtools.__path__.append(self.package.base)