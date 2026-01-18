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
def test_signalHandlersInstalledDuringStartup(self) -> None:
    """
        Signal handlers are installed in responsed to the C{"during"}
        C{"startup"}.
        """
    reactor = self.buildReactor()
    phase: Optional[str] = None

    def beforeStartup() -> None:
        nonlocal phase
        phase = 'before'

    def afterStartup() -> None:
        nonlocal phase
        phase = 'after'
    reactor.addSystemEventTrigger('before', 'startup', beforeStartup)
    reactor.addSystemEventTrigger('after', 'startup', afterStartup)
    sawPhase = []

    def fakeSignal(signum: int, action: Callable[[int, FrameType], None]) -> None:
        sawPhase.append(phase)
    testCase = cast(SynchronousTestCase, self)
    testCase.patch(signal, 'signal', fakeSignal)
    reactor.callWhenRunning(reactor.stop)
    testCase.assertIsNone(phase)
    testCase.assertEqual(sawPhase, [])
    self.runReactor(reactor)
    testCase.assertIn('before', sawPhase)
    testCase.assertEqual(phase, 'after')