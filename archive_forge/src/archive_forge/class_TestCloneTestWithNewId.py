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
class TestCloneTestWithNewId(TestCase):
    """Tests for clone_test_with_new_id."""
    run_test_with = FullStackRunTest

    def test_clone_test_with_new_id(self):

        class FooTestCase(TestCase):

            def test_foo(self):
                pass
        test = FooTestCase('test_foo')
        oldName = test.id()
        newName = self.getUniqueString()
        newTest = clone_test_with_new_id(test, newName)
        self.assertEqual(newName, newTest.id())
        self.assertEqual(oldName, test.id(), 'the original test instance should be unchanged.')

    def test_cloned_testcase_does_not_share_details(self):
        """A cloned TestCase does not share the details dict."""

        class Test(TestCase):

            def test_foo(self):
                self.addDetail('foo', content.Content('text/plain', lambda: 'foo'))
        orig_test = Test('test_foo')
        cloned_test = clone_test_with_new_id(orig_test, self.getUniqueString())
        orig_test.run(unittest.TestResult())
        self.assertEqual('foo', orig_test.getDetails()['foo'].iter_bytes())
        self.assertEqual(None, cloned_test.getDetails().get('foo'))