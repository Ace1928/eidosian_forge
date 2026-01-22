from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestTestCaseSuper(TestCase):
    run_test_with = FullStackRunTest

    def test_setup_uses_super(self):

        class OtherBaseCase(unittest.TestCase):
            setup_called = False

            def setUp(self):
                self.setup_called = True
                super().setUp()

        class OurCase(TestCase, OtherBaseCase):

            def runTest(self):
                pass
        test = OurCase()
        test.setUp()
        test.tearDown()
        self.assertTrue(test.setup_called)

    def test_teardown_uses_super(self):

        class OtherBaseCase(unittest.TestCase):
            teardown_called = False

            def tearDown(self):
                self.teardown_called = True
                super().tearDown()

        class OurCase(TestCase, OtherBaseCase):

            def runTest(self):
                pass
        test = OurCase()
        test.setUp()
        test.tearDown()
        self.assertTrue(test.teardown_called)