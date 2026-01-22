from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
class EventAsTextTests(unittest.TestCase):
    """
    Tests for L{eventAsText}, all of which ensure that the
    returned type is UTF-8 decoded text.
    """

    def test_eventWithTraceback(self) -> None:
        """
        An event with a C{log_failure} key will have a traceback appended.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        event: LogEvent = {'log_format': 'This is a test log message'}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIn(str(f.getTraceback()), eventText)
        self.assertIn('This is a test log message', eventText)

    def test_formatEmptyEventWithTraceback(self) -> None:
        """
        An event with an empty C{log_format} key appends a traceback from
        the accompanying failure.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        event: LogEvent = {'log_format': ''}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIn(str(f.getTraceback()), eventText)
        self.assertIn('This is a fake error', eventText)

    def test_formatUnformattableWithTraceback(self) -> None:
        """
        An event with an unformattable value in the C{log_format} key still
        has a traceback appended.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        event = {'log_format': '{evil()}', 'evil': lambda: 1 / 0}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIsInstance(eventText, str)
        self.assertIn(str(f.getTraceback()), eventText)
        self.assertIn('This is a fake error', eventText)

    def test_formatUnformattableErrorWithTraceback(self) -> None:
        """
        An event with an unformattable value in the C{log_format} key, that
        throws an exception when __repr__ is invoked still has a traceback
        appended.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        event: LogEvent = {'log_format': '{evil()}', 'evil': lambda: 1 / 0, cast(str, Unformattable()): 'gurk'}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIsInstance(eventText, str)
        self.assertIn('MESSAGE LOST', eventText)
        self.assertIn(str(f.getTraceback()), eventText)
        self.assertIn('This is a fake error', eventText)

    def test_formatEventUnformattableTraceback(self) -> None:
        """
        If a traceback cannot be appended, a message indicating this is true
        is appended.
        """
        event: LogEvent = {'log_format': ''}
        event['log_failure'] = object()
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIsInstance(eventText, str)
        self.assertIn('(UNABLE TO OBTAIN TRACEBACK FROM EVENT)', eventText)

    def test_formatEventNonCritical(self) -> None:
        """
        An event with no C{log_failure} key will not have a traceback appended.
        """
        event: LogEvent = {'log_format': 'This is a test log message'}
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIsInstance(eventText, str)
        self.assertIn('This is a test log message', eventText)

    def test_formatTracebackMultibyte(self) -> None:
        """
        An exception message with multibyte characters is properly handled.
        """
        try:
            raise CapturedError('€')
        except CapturedError:
            f = Failure()
        event: LogEvent = {'log_format': 'This is a test log message'}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIn('€', eventText)
        self.assertIn('Traceback', eventText)

    def test_formatTracebackHandlesUTF8DecodeFailure(self) -> None:
        """
        An error raised attempting to decode the UTF still produces a
        valid log message.
        """
        try:
            raise CapturedError(b'\xff\xfet\x00e\x00s\x00t\x00')
        except CapturedError:
            f = Failure()
        event: LogEvent = {'log_format': 'This is a test log message'}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
        self.assertIn('Traceback', eventText)
        self.assertIn('CapturedError(b"\\xff\\xfet\\x00e\\x00s\\x00t\\x00")', eventText)

    def test_eventAsTextSystemOnly(self) -> None:
        """
        If includeSystem is specified as the only option no timestamp or
        traceback are printed.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event: LogEvent = {'log_format': 'ABCD', 'log_system': 'fake_system', 'log_time': t}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=False, includeTraceback=False, includeSystem=True)
        self.assertEqual(eventText, '[fake_system] ABCD')

    def test_eventAsTextTimestampOnly(self) -> None:
        """
        If includeTimestamp is specified as the only option no system or
        traceback are printed.
        """
        if tzset is None:
            raise SkipTest('Platform cannot change timezone; unable to verify offsets.')
        addTZCleanup(self)
        setTZ('UTC+00')
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event: LogEvent = {'log_format': 'ABCD', 'log_system': 'fake_system', 'log_time': t}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=True, includeTraceback=False, includeSystem=False)
        self.assertEqual(eventText, '2013-09-24T11:40:47+0000 ABCD')

    def test_eventAsTextSystemMissing(self) -> None:
        """
        If includeSystem is specified with a missing system [-#-]
        is used.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event: LogEvent = {'log_format': 'ABCD', 'log_time': t}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=False, includeTraceback=False, includeSystem=True)
        self.assertEqual(eventText, '[-#-] ABCD')

    def test_eventAsTextSystemMissingNamespaceAndLevel(self) -> None:
        """
        If includeSystem is specified with a missing system but
        namespace and level are present they are used.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event: LogEvent = {'log_format': 'ABCD', 'log_time': t, 'log_level': LogLevel.info, 'log_namespace': 'test'}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=False, includeTraceback=False, includeSystem=True)
        self.assertEqual(eventText, '[test#info] ABCD')

    def test_eventAsTextSystemMissingLevelOnly(self) -> None:
        """
        If includeSystem is specified with a missing system but
        level is present, level is included.
        """
        try:
            raise CapturedError('This is a fake error')
        except CapturedError:
            f = Failure()
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event: LogEvent = {'log_format': 'ABCD', 'log_time': t, 'log_level': LogLevel.info}
        event['log_failure'] = f
        eventText = eventAsText(event, includeTimestamp=False, includeTraceback=False, includeSystem=True)
        self.assertEqual(eventText, '[-#info] ABCD')