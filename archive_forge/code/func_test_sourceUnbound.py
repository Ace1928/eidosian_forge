from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_sourceUnbound(self) -> None:
    """
        C{log_source} event key is L{None}.
        """

    @implementer(ILogObserver)
    def observer(event: LogEvent) -> None:
        self.assertIsNone(event['log_source'])
    log = TestLogger(observer=observer)
    log.info()