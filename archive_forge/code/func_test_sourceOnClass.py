from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_sourceOnClass(self) -> None:
    """
        C{log_source} event key refers to the class.
        """

    @implementer(ILogObserver)
    def observer(event: LogEvent) -> None:
        self.assertEqual(event['log_source'], Thingo)

    class Thingo:
        log = TestLogger(observer=observer)
    cast(TestLogger, Thingo.log).info()