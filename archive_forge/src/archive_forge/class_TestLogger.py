from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
class TestLogger(Logger):
    """
    L{Logger} with an overridden C{emit} method that keeps track of received
    events.
    """

    def emit(self, level: NamedConstant, format: Optional[str]=None, **kwargs: object) -> None:

        @implementer(ILogObserver)
        def observer(event: LogEvent) -> None:
            self.event = event
        globalLogPublisher.addObserver(observer)
        try:
            Logger.emit(self, level, format, **kwargs)
        finally:
            globalLogPublisher.removeObserver(observer)
        self.emitted = {'level': level, 'format': format, 'kwargs': kwargs}