import io
from typing import IO, Any, List, Optional, TextIO, Tuple, Type, cast
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._file import textFileLogObserver
from .._global import MORE_THAN_ONCE_WARNING, LogBeginner
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
from ..test.test_stdlib import nextLine
def test_warningsModule(self) -> None:
    """
        L{LogBeginner.beginLoggingTo} will redirect the warnings of its
        warnings module into the logging system.
        """
    self.warningsModule.showwarning('a message', DeprecationWarning, __file__, 1)
    events: List[LogEvent] = []
    self.beginner.beginLoggingTo([cast(ILogObserver, events.append)])
    self.warningsModule.showwarning('another message', DeprecationWarning, __file__, 2)
    f = io.StringIO()
    self.warningsModule.showwarning('yet another', DeprecationWarning, __file__, 3, file=f)
    self.assertEqual(self.warningsModule.warnings, [('a message', DeprecationWarning, __file__, 1, None, None), ('yet another', DeprecationWarning, __file__, 3, f, None)])
    compareEvents(self, events, [dict(warning='another message', category=DeprecationWarning.__module__ + '.' + DeprecationWarning.__name__, filename=__file__, lineno=2)])