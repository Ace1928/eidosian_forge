import datetime
import io
import os
import tempfile
import unittest
from io import BytesIO
from testtools import PlaceHolder, TestCase, TestResult, skipIf
from testtools.compat import _b, _u
from testtools.content import Content, TracebackContent, text_content
from testtools.content_type import ContentType
from testtools.matchers import Contains, Equals, MatchesAny
import iso8601
import subunit
from subunit.tests import (_remote_exception_repr,
class TestRemotedTestCase(unittest.TestCase):

    def test_simple(self):
        test = subunit.RemotedTestCase('A test description')
        self.assertRaises(NotImplementedError, test.setUp)
        self.assertRaises(NotImplementedError, test.tearDown)
        self.assertEqual('A test description', test.shortDescription())
        self.assertEqual('A test description', test.id())
        self.assertEqual('A test description (subunit.RemotedTestCase)', '%s' % test)
        self.assertEqual("<subunit.RemotedTestCase description='A test description'>", '%r' % test)
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual([(test, _remote_exception_repr + ': ' + 'Cannot run RemotedTestCases.\n\n')], result.errors)
        self.assertEqual(1, result.testsRun)
        another_test = subunit.RemotedTestCase('A test description')
        self.assertEqual(test, another_test)
        different_test = subunit.RemotedTestCase('ofo')
        self.assertNotEqual(test, different_test)
        self.assertNotEqual(another_test, different_test)