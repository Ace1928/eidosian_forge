from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
def test_observerRaises(self) -> None:
    """
        Observer raises an exception during fan out: a failure is logged, but
        not re-raised.  Life goes on.
        """
    event = dict(foo=1, bar=2)
    exception = RuntimeError('ARGH! EVIL DEATH!')
    events: List[LogEvent] = []

    @implementer(ILogObserver)
    def observer(event: LogEvent) -> None:
        shouldRaise = not events
        events.append(event)
        if shouldRaise:
            raise exception
    collector: List[LogEvent] = []
    publisher = LogPublisher(observer, cast(ILogObserver, collector.append))
    publisher(event)
    self.assertIn(event, events)
    errors = [e['log_failure'] for e in collector if 'log_failure' in e]
    self.assertEqual(len(errors), 1)
    self.assertIs(errors[0].value, exception)
    self.assertEqual(len(events), 1)