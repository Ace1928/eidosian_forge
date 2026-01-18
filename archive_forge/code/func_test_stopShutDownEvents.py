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
def test_stopShutDownEvents(self) -> None:
    """
        C{reactor.stop()} fires all three phases of shutdown event triggers
        before it makes C{reactor.run()} return.
        """
    reactor = self.buildReactor()
    events = []
    reactor.addSystemEventTrigger('before', 'shutdown', lambda: events.append(('before', 'shutdown')))
    reactor.addSystemEventTrigger('during', 'shutdown', lambda: events.append(('during', 'shutdown')))
    reactor.addSystemEventTrigger('after', 'shutdown', lambda: events.append(('after', 'shutdown')))
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    cast(SynchronousTestCase, self).assertEqual(events, [('before', 'shutdown'), ('during', 'shutdown'), ('after', 'shutdown')])