import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_restores_observers(self):
    from testtools.twistedsupport._runtest import run_with_log_observers
    from twisted.python import log
    log.addObserver(lambda *args: None)
    observers = list(log.theLogPublisher.observers)
    run_with_log_observers([], lambda: None)
    self.assertEqual(observers, log.theLogPublisher.observers)