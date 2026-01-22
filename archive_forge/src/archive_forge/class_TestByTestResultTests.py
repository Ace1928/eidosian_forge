import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
class TestByTestResultTests(testtools.TestCase):

    def setUp(self):
        super().setUp()
        self.log = []
        self.result = subunit.test_results.TestByTestResult(self.on_test)
        self.result._now = iter(range(5)).__next__

    def assertCalled(self, **kwargs):
        defaults = {'test': self, 'tags': set(), 'details': None, 'start_time': 0, 'stop_time': 1}
        defaults.update(kwargs)
        self.assertEqual([defaults], self.log)

    def on_test(self, **kwargs):
        self.log.append(kwargs)

    def test_no_tests_nothing_reported(self):
        self.result.startTestRun()
        self.result.stopTestRun()
        self.assertEqual([], self.log)

    def test_add_success(self):
        self.result.startTest(self)
        self.result.addSuccess(self)
        self.result.stopTest(self)
        self.assertCalled(status='success')

    def test_add_success_details(self):
        self.result.startTest(self)
        details = {'foo': 'bar'}
        self.result.addSuccess(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='success', details=details)

    def test_tags(self):
        if not getattr(self.result, 'tags', None):
            self.skipTest('No tags in testtools')
        self.result.tags(['foo'], [])
        self.result.startTest(self)
        self.result.addSuccess(self)
        self.result.stopTest(self)
        self.assertCalled(status='success', tags={'foo'})

    def test_add_error(self):
        self.result.startTest(self)
        try:
            1 / 0
        except ZeroDivisionError:
            error = sys.exc_info()
        self.result.addError(self, error)
        self.result.stopTest(self)
        self.assertCalled(status='error', details={'traceback': TracebackContent(error, self)})

    def test_add_error_details(self):
        self.result.startTest(self)
        details = {'foo': text_content('bar')}
        self.result.addError(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='error', details=details)

    def test_add_failure(self):
        self.result.startTest(self)
        try:
            self.fail('intentional failure')
        except self.failureException:
            failure = sys.exc_info()
        self.result.addFailure(self, failure)
        self.result.stopTest(self)
        self.assertCalled(status='failure', details={'traceback': TracebackContent(failure, self)})

    def test_add_failure_details(self):
        self.result.startTest(self)
        details = {'foo': text_content('bar')}
        self.result.addFailure(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='failure', details=details)

    def test_add_xfail(self):
        self.result.startTest(self)
        try:
            1 / 0
        except ZeroDivisionError:
            error = sys.exc_info()
        self.result.addExpectedFailure(self, error)
        self.result.stopTest(self)
        self.assertCalled(status='xfail', details={'traceback': TracebackContent(error, self)})

    def test_add_xfail_details(self):
        self.result.startTest(self)
        details = {'foo': text_content('bar')}
        self.result.addExpectedFailure(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='xfail', details=details)

    def test_add_unexpected_success(self):
        self.result.startTest(self)
        details = {'foo': 'bar'}
        self.result.addUnexpectedSuccess(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='success', details=details)

    def test_add_skip_reason(self):
        self.result.startTest(self)
        reason = self.getUniqueString()
        self.result.addSkip(self, reason)
        self.result.stopTest(self)
        self.assertCalled(status='skip', details={'reason': text_content(reason)})

    def test_add_skip_details(self):
        self.result.startTest(self)
        details = {'foo': 'bar'}
        self.result.addSkip(self, details=details)
        self.result.stopTest(self)
        self.assertCalled(status='skip', details=details)

    def test_twice(self):
        self.result.startTest(self)
        self.result.addSuccess(self, details={'foo': 'bar'})
        self.result.stopTest(self)
        self.result.startTest(self)
        self.result.addSuccess(self)
        self.result.stopTest(self)
        self.assertEqual([{'test': self, 'status': 'success', 'start_time': 0, 'stop_time': 1, 'tags': set(), 'details': {'foo': 'bar'}}, {'test': self, 'status': 'success', 'start_time': 2, 'stop_time': 3, 'tags': set(), 'details': None}], self.log)