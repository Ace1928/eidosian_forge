import logging as stdlibLogging
from typing import Mapping, Tuple
from zope.interface import implementer
from constantly import NamedConstant
from twisted.python.compat import currentframe
from ._format import formatEvent
from ._interfaces import ILogObserver, LogEvent
from ._levels import LogLevel
fromStdlibLogLevelMapping = _reverseLogLevelMapping()
def _reverseLogLevelMapping() -> Mapping[int, NamedConstant]:
    """
    Reverse the above mapping, adding both the numerical keys used above and
    the corresponding string keys also used by python logging.
    @return: the reversed mapping
    """
    mapping = {}
    for logLevel, pyLogLevel in toStdlibLogLevelMapping.items():
        mapping[pyLogLevel] = logLevel
        mapping[stdlibLogging.getLevelName(pyLogLevel)] = logLevel
    return mapping