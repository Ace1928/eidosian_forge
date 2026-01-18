import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_timeout_causes_test_error(self):

    class SomeCase(TestCase):

        def test_pause(self):
            return defer.Deferred()
    test = SomeCase('test_pause')
    runner = self.make_runner(test)
    result = self.make_result()
    runner.run(result)
    error = result._events[1][2]
    self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
    self.assertIn('TimeoutError', str(error['traceback']))