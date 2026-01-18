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
def test_runWithAsynchronousBeforeStartupTrigger(self) -> None:
    """
        When there is a C{'before'} C{'startup'} trigger which returns an
        unfired L{Deferred}, C{reactor.run()} starts the reactor and does not
        return until after C{reactor.stop()} is called
        """
    events = []

    def trigger() -> Deferred[object]:
        events.append('trigger')
        d: Deferred[object] = Deferred()
        d.addCallback(callback)
        reactor.callLater(0, d.callback, None)
        return d

    def callback(ignored: object) -> None:
        events.append('callback')
        reactor.stop()
    reactor = self.buildReactor()
    reactor.addSystemEventTrigger('before', 'startup', trigger)
    self.runReactor(reactor)
    cast(SynchronousTestCase, self).assertEqual(events, ['trigger', 'callback'])