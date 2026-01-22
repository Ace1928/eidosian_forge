import json
from itertools import count
from typing import Any, Callable, Optional
from twisted.trial import unittest
from .._flatten import KeyFlattener, aFormatter, extractField, flattenEvent
from .._format import formatEvent
from .._interfaces import LogEvent
def _test_formatFlatEvent_fieldNamesSame(self, event: Optional[LogEvent]=None) -> LogEvent:
    """
        The same format field used twice in one event is rendered twice.

        @param event: An event to flatten.  If L{None}, create a new event.
        @return: C{event} or the event created.
        """
    if event is None:
        counter = count()

        class CountStr:
            """
                Hack
                """

            def __str__(self) -> str:
                return str(next(counter))
        event = dict(log_format='{x} {x}', x=CountStr())
    flattenEvent(event)
    self.assertEqual(formatEvent(event), '0 1')
    return event