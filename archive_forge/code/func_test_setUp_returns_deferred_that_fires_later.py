import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_setUp_returns_deferred_that_fires_later(self):
    call_log = []
    marker = object()
    d = defer.Deferred().addCallback(call_log.append)

    class SomeCase(TestCase):

        def setUp(self):
            super().setUp()
            call_log.append('setUp')
            return d

        def test_something(self):
            call_log.append('test')

    def fire_deferred():
        self.assertThat(call_log, Equals(['setUp']))
        d.callback(marker)
    test = SomeCase('test_something')
    timeout = self.make_timeout()
    runner = self.make_runner(test, timeout=timeout)
    result = self.make_result()
    reactor = self.make_reactor()
    reactor.callLater(timeout, fire_deferred)
    runner.run(result)
    self.assertThat(call_log, Equals(['setUp', marker, 'test']))