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
def test_runAfterCrash(self) -> None:
    """
        C{reactor.run()} restarts the reactor after it has been stopped by
        C{reactor.crash()}.
        """
    events: List[Union[str, Tuple[str, bool]]] = []

    def crash() -> None:
        events.append('crash')
        reactor.crash()
    reactor = self.buildReactor()
    reactor.callWhenRunning(crash)
    self.runReactor(reactor)

    def stop() -> None:
        events.append(('stop', reactor.running))
        reactor.stop()
    reactor.callWhenRunning(stop)
    self.runReactor(reactor)
    cast(SynchronousTestCase, self).assertEqual(events, ['crash', ('stop', True)])