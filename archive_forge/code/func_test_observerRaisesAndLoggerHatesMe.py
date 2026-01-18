from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
def test_observerRaisesAndLoggerHatesMe(self) -> None:
    """
        Observer raises an exception during fan out and the publisher's Logger
        pukes when the failure is reported.  The exception does not propagate
        back to the caller.
        """
    event = dict(foo=1, bar=2)
    exception = RuntimeError('ARGH! EVIL DEATH!')

    @implementer(ILogObserver)
    def observer(event: LogEvent) -> None:
        raise RuntimeError('Sad panda')

    class GurkLogger(Logger):

        def failure(self, *args: object, **kwargs: object) -> None:
            raise exception
    publisher = LogPublisher(observer)
    publisher.log = GurkLogger()
    publisher(event)