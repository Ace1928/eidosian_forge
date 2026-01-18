import signal
import time
from types import FrameType
from typing import Callable, List, Optional, Tuple, Union, cast
from twisted.internet.abstract import FileDescriptor
from twisted.internet.defer import Deferred
from twisted.internet.error import ReactorAlreadyRunning, ReactorNotRestartable
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_runningForStartupEvents(self) -> None:
    """
        The reactor is not running when C{"before"} C{"startup"} triggers are
        called and is running when C{"during"} and C{"after"} C{"startup"}
        triggers are called.
        """
    reactor = self.buildReactor()
    state = {}

    def beforeStartup() -> None:
        state['before'] = reactor.running

    def duringStartup() -> None:
        state['during'] = reactor.running

    def afterStartup() -> None:
        state['after'] = reactor.running
    testCase = cast(SynchronousTestCase, self)
    reactor.addSystemEventTrigger('before', 'startup', beforeStartup)
    reactor.addSystemEventTrigger('during', 'startup', duringStartup)
    reactor.addSystemEventTrigger('after', 'startup', afterStartup)
    reactor.callWhenRunning(reactor.stop)
    testCase.assertEqual(state, {})
    self.runReactor(reactor)
    testCase.assertEqual(state, {'before': False, 'during': True, 'after': True})