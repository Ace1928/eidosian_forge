import logging as stdlibLogging
from typing import Mapping, Tuple
from zope.interface import implementer
from constantly import NamedConstant
from twisted.python.compat import currentframe
from ._format import formatEvent
from ._interfaces import ILogObserver, LogEvent
from ._levels import LogLevel
fromStdlibLogLevelMapping = _reverseLogLevelMapping()
class StringifiableFromEvent:
    """
    An object that implements C{__str__()} in order to defer the work of
    formatting until it's converted into a C{str}.
    """

    def __init__(self, event: LogEvent) -> None:
        """
        @param event: An event.
        """
        self.event = event

    def __str__(self) -> str:
        return formatEvent(self.event)

    def __bytes__(self) -> bytes:
        return str(self).encode('utf-8')