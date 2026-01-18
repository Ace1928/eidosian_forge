import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_restore_observers_after_timeout(self):
    publisher, observers = _get_global_publisher_and_observers()

    class LogSomething(TestCase):

        def test_something(self):
            return defer.Deferred()
    test = LogSomething('test_something')
    runner = self.make_runner(test, timeout=0.0001)
    result = self.make_result()
    runner.run(result)
    self.assertThat(_get_global_publisher_and_observers()[1], Equals(observers))