import logging as stdlibLogging
from typing import Mapping, Tuple
from zope.interface import implementer
from constantly import NamedConstant
from twisted.python.compat import currentframe
from ._format import formatEvent
from ._interfaces import ILogObserver, LogEvent
from ._levels import LogLevel
fromStdlibLogLevelMapping = _reverseLogLevelMapping()
def _findCaller(self, stackInfo: bool=False, stackLevel: int=1) -> Tuple[str, int, str, None]:
    """
        Based on the stack depth passed to this L{STDLibLogObserver}, identify
        the calling function.

        @param stackInfo: Whether or not to construct stack information.
            (Currently ignored.)
        @param stackLevel: The number of stack frames to skip when determining
            the caller (currently ignored; use stackDepth on the instance).

        @return: Depending on Python version, either a 3-tuple of (filename,
            lineno, name) or a 4-tuple of that plus stack information.
        """
    f = currentframe(self.stackDepth)
    co = f.f_code
    extra = (None,)
    return (co.co_filename, f.f_lineno, co.co_name) + extra