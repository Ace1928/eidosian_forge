import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
class LegacyLogObserverWrapperTests(unittest.TestCase):
    """
    Tests for L{LegacyLogObserverWrapper}.
    """

    def test_interface(self) -> None:
        """
        L{LegacyLogObserverWrapper} is an L{ILogObserver}.
        """
        legacyObserver = cast(legacyLog.ILogObserver, lambda e: None)
        observer = LegacyLogObserverWrapper(legacyObserver)
        try:
            verifyObject(ILogObserver, observer)
        except BrokenMethodImplementation as e:
            self.fail(e)

    def test_repr(self) -> None:
        """
        L{LegacyLogObserverWrapper} returns the expected string.
        """

        @implementer(legacyLog.ILogObserver)
        class LegacyObserver:

            def __repr__(self) -> str:
                return '<Legacy Observer>'

            def __call__(self, eventDict: legacyLog.EventDict) -> None:
                return
        observer = LegacyLogObserverWrapper(LegacyObserver())
        self.assertEqual(repr(observer), 'LegacyLogObserverWrapper(<Legacy Observer>)')

    def observe(self, event: LogEvent) -> LogEvent:
        """
        Send an event to a wrapped legacy observer and capture the event as
        seen by that observer.

        @param event: an event

        @return: the event as observed by the legacy wrapper
        """
        events: List[LogEvent] = []
        legacyObserver = cast(legacyLog.ILogObserver, lambda e: events.append(e))
        observer = LegacyLogObserverWrapper(legacyObserver)
        observer(event)
        self.assertEqual(len(events), 1)
        return events[0]

    def forwardAndVerify(self, event: LogEvent) -> LogEvent:
        """
        Send an event to a wrapped legacy observer and verify that its data is
        preserved.

        @param event: an event

        @return: the event as observed by the legacy wrapper
        """
        event.setdefault('log_time', time())
        event.setdefault('log_system', '-')
        event.setdefault('log_level', LogLevel.info)
        observed = self.observe(dict(event))
        for key, value in event.items():
            self.assertIn(key, observed)
        return observed

    def test_forward(self) -> None:
        """
        Basic forwarding: event keys as observed by a legacy observer are the
        same.
        """
        self.forwardAndVerify(dict(foo=1, bar=2))

    def test_time(self) -> None:
        """
        The new-style C{"log_time"} key is copied to the old-style C{"time"}
        key.
        """
        stamp = time()
        event = self.forwardAndVerify(dict(log_time=stamp))
        self.assertEqual(event['time'], stamp)

    def test_timeAlreadySet(self) -> None:
        """
        The new-style C{"log_time"} key does not step on a pre-existing
        old-style C{"time"} key.
        """
        stamp = time()
        event = self.forwardAndVerify(dict(log_time=stamp + 1, time=stamp))
        self.assertEqual(event['time'], stamp)

    def test_system(self) -> None:
        """
        The new-style C{"log_system"} key is copied to the old-style
        C{"system"} key.
        """
        event = self.forwardAndVerify(dict(log_system='foo'))
        self.assertEqual(event['system'], 'foo')

    def test_systemAlreadySet(self) -> None:
        """
        The new-style C{"log_system"} key does not step on a pre-existing
        old-style C{"system"} key.
        """
        event = self.forwardAndVerify(dict(log_system='foo', system='bar'))
        self.assertEqual(event['system'], 'bar')

    def test_noSystem(self) -> None:
        """
        If the new-style C{"log_system"} key is absent, the old-style
        C{"system"} key is set to C{"-"}.
        """
        event = dict(log_time=time(), log_level=LogLevel.info)
        observed = self.observe(dict(event))
        self.assertEqual(observed['system'], '-')

    def test_levelNotChange(self) -> None:
        """
        If explicitly set, the C{isError} key will be preserved when forwarding
        from a new-style logging emitter to a legacy logging observer,
        regardless of log level.
        """
        self.forwardAndVerify(dict(log_level=LogLevel.info, isError=1))
        self.forwardAndVerify(dict(log_level=LogLevel.warn, isError=1))
        self.forwardAndVerify(dict(log_level=LogLevel.error, isError=0))
        self.forwardAndVerify(dict(log_level=LogLevel.critical, isError=0))

    def test_pythonLogLevelNotSet(self) -> None:
        """
        The new-style C{"log_level"} key is not translated to the old-style
        C{"logLevel"} key.

        Events are forwarded from the old module from to new module and are
        then seen by old-style observers.
        We don't want to add unexpected keys to old-style events.
        """
        event = self.forwardAndVerify(dict(log_level=LogLevel.info))
        self.assertNotIn('logLevel', event)

    def test_stringPythonLogLevel(self) -> None:
        """
        If a stdlib log level was provided as a string (eg. C{"WARNING"}) in
        the legacy "logLevel" key, it does not get converted to a number.
        The documentation suggested that numerical values should be used but
        this was not a requirement.
        """
        event = self.forwardAndVerify(dict(logLevel='WARNING'))
        self.assertEqual(event['logLevel'], 'WARNING')

    def test_message(self) -> None:
        """
        The old-style C{"message"} key is added, even if no new-style
        C{"log_format"} is given, as it is required, but may be empty.
        """
        event = self.forwardAndVerify(dict())
        self.assertEqual(event['message'], ())

    def test_messageAlreadySet(self) -> None:
        """
        The old-style C{"message"} key is not modified if it already exists.
        """
        event = self.forwardAndVerify(dict(message=('foo', 'bar')))
        self.assertEqual(event['message'], ('foo', 'bar'))

    def test_format(self) -> None:
        """
        Formatting is translated such that text is rendered correctly, even
        though old-style logging doesn't use PEP 3101 formatting.
        """
        event = self.forwardAndVerify(dict(log_format='Hello, {who}!', who='world'))
        self.assertEqual(legacyLog.textFromEventDict(event), 'Hello, world!')

    def test_formatMessage(self) -> None:
        """
        Using the message key, which is special in old-style, works for
        new-style formatting.
        """
        event = self.forwardAndVerify(dict(log_format='Hello, {message}!', message='world'))
        self.assertEqual(legacyLog.textFromEventDict(event), 'Hello, world!')

    def test_formatAlreadySet(self) -> None:
        """
        Formatting is not altered if the old-style C{"format"} key already
        exists.
        """
        event = self.forwardAndVerify(dict(log_format='Hello!', format='Howdy!'))
        self.assertEqual(legacyLog.textFromEventDict(event), 'Howdy!')

    def eventWithFailure(self, **values: object) -> LogEvent:
        """
        Create a new-style event with a captured failure.

        @param values: Additional values to include in the event.

        @return: the new event
        """
        failure = Failure(RuntimeError('nyargh!'))
        return self.forwardAndVerify(dict(log_failure=failure, log_format='oopsie...', **values))

    def test_failure(self) -> None:
        """
        Captured failures in the new style set the old-style C{"failure"},
        C{"isError"}, and C{"why"} keys.
        """
        event = self.eventWithFailure()
        self.assertIs(event['failure'], event['log_failure'])
        self.assertTrue(event['isError'])
        self.assertEqual(event['why'], 'oopsie...')

    def test_failureAlreadySet(self) -> None:
        """
        Captured failures in the new style do not step on a pre-existing
        old-style C{"failure"} key.
        """
        failure = Failure(RuntimeError('Weak salsa!'))
        event = self.eventWithFailure(failure=failure)
        self.assertIs(event['failure'], failure)

    def test_isErrorAlreadySet(self) -> None:
        """
        Captured failures in the new style do not step on a pre-existing
        old-style C{"isError"} key.
        """
        event = self.eventWithFailure(isError=0)
        self.assertEqual(event['isError'], 0)

    def test_whyAlreadySet(self) -> None:
        """
        Captured failures in the new style do not step on a pre-existing
        old-style C{"failure"} key.
        """
        event = self.eventWithFailure(why='blah')
        self.assertEqual(event['why'], 'blah')