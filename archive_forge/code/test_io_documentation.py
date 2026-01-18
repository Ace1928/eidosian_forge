import sys
from typing import List, Optional
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._io import LoggingFile
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher

        Construct a L{LoggingFile} with a built-in observer.

        @param level: C{level} argument to L{LoggingFile}
        @param encoding: C{encoding} argument to L{LoggingFile}

        @return: a L{TestLoggingFile} with an observer that appends received
            events into the file's C{events} attribute (a L{list}) and
            event messages into the file's C{messages} attribute (a L{list}).
        