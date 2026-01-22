import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
class TestSynchronousDeferredRunTest(NeedsTwistedTestCase):

    def make_result(self):
        return ExtendedTestResult()

    def make_runner(self, test):
        return SynchronousDeferredRunTest(test, test.exception_handlers)

    def test_success(self):

        class SomeCase(TestCase):

            def test_success(self):
                return defer.succeed(None)
        test = SomeCase('test_success')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat(result._events, Equals([('startTest', test), ('addSuccess', test), ('stopTest', test)]))

    def test_failure(self):

        class SomeCase(TestCase):

            def test_failure(self):
                return defer.maybeDeferred(self.fail, 'Egads!')
        test = SomeCase('test_failure')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addFailure', test), ('stopTest', test)]))

    def test_setUp_followed_by_test(self):

        class SomeCase(TestCase):

            def setUp(self):
                super().setUp()
                return defer.succeed(None)

            def test_failure(self):
                return defer.maybeDeferred(self.fail, 'Egads!')
        test = SomeCase('test_failure')
        runner = self.make_runner(test)
        result = self.make_result()
        runner.run(result)
        self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addFailure', test), ('stopTest', test)]))