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
class TestTestProtocolServerPassThrough(unittest.TestCase):

    def setUp(self):
        self.stdout = BytesIO()
        self.test = subunit.RemotedTestCase('old mcdonald')
        self.client = ExtendedTestResult()
        self.protocol = subunit.TestProtocolServer(self.client, self.stdout)

    def keywords_before_test(self):
        self.protocol.lineReceived(_b('failure a\n'))
        self.protocol.lineReceived(_b('failure: a\n'))
        self.protocol.lineReceived(_b('error a\n'))
        self.protocol.lineReceived(_b('error: a\n'))
        self.protocol.lineReceived(_b('success a\n'))
        self.protocol.lineReceived(_b('success: a\n'))
        self.protocol.lineReceived(_b('successful a\n'))
        self.protocol.lineReceived(_b('successful: a\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.assertEqual(self.stdout.getvalue(), _b('failure a\nfailure: a\nerror a\nerror: a\nsuccess a\nsuccess: a\nsuccessful a\nsuccessful: a\n]\n'))

    def test_keywords_before_test(self):
        self.keywords_before_test()
        self.assertEqual(self.client._events, [])

    def test_keywords_after_error(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('error old mcdonald\n'))
        self.keywords_before_test()
        self.assertEqual([('startTest', self.test), ('addError', self.test, {}), ('stopTest', self.test)], self.client._events)

    def test_keywords_after_failure(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('failure old mcdonald\n'))
        self.keywords_before_test()
        self.assertEqual(self.client._events, [('startTest', self.test), ('addFailure', self.test, {}), ('stopTest', self.test)])

    def test_keywords_after_success(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('success old mcdonald\n'))
        self.keywords_before_test()
        self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)

    def test_keywords_after_test(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('failure a\n'))
        self.protocol.lineReceived(_b('failure: a\n'))
        self.protocol.lineReceived(_b('error a\n'))
        self.protocol.lineReceived(_b('error: a\n'))
        self.protocol.lineReceived(_b('success a\n'))
        self.protocol.lineReceived(_b('success: a\n'))
        self.protocol.lineReceived(_b('successful a\n'))
        self.protocol.lineReceived(_b('successful: a\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.protocol.lineReceived(_b('failure old mcdonald\n'))
        self.assertEqual(self.stdout.getvalue(), _b('test old mcdonald\nfailure a\nfailure: a\nerror a\nerror: a\nsuccess a\nsuccess: a\nsuccessful a\nsuccessful: a\n]\n'))
        self.assertEqual(self.client._events, [('startTest', self.test), ('addFailure', self.test, {}), ('stopTest', self.test)])

    def test_keywords_during_failure(self):
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('failure: old mcdonald [\n'))
        self.protocol.lineReceived(_b('test old mcdonald\n'))
        self.protocol.lineReceived(_b('failure a\n'))
        self.protocol.lineReceived(_b('failure: a\n'))
        self.protocol.lineReceived(_b('error a\n'))
        self.protocol.lineReceived(_b('error: a\n'))
        self.protocol.lineReceived(_b('success a\n'))
        self.protocol.lineReceived(_b('success: a\n'))
        self.protocol.lineReceived(_b('successful a\n'))
        self.protocol.lineReceived(_b('successful: a\n'))
        self.protocol.lineReceived(_b(' ]\n'))
        self.protocol.lineReceived(_b(']\n'))
        self.assertEqual(self.stdout.getvalue(), _b(''))
        details = {}
        details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b('test old mcdonald\nfailure a\nfailure: a\nerror a\nerror: a\nsuccess a\nsuccess: a\nsuccessful a\nsuccessful: a\n]\n')])
        self.assertEqual(self.client._events, [('startTest', self.test), ('addFailure', self.test, details), ('stopTest', self.test)])

    def test_stdout_passthrough(self):
        """Lines received which cannot be interpreted as any protocol action
        should be passed through to sys.stdout.
        """
        bytes = _b('randombytes\n')
        self.protocol.lineReceived(bytes)
        self.assertEqual(self.stdout.getvalue(), bytes)