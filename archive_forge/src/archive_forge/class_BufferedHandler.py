from __future__ import annotations
import logging as py_logging
import sys
from inspect import getsourcefile
from io import BytesIO, TextIOWrapper
from logging import Formatter, LogRecord, StreamHandler, getLogger
from typing import List, Optional, Tuple
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._stdlib import STDLibLogObserver
class BufferedHandler(py_logging.Handler):
    """
    A L{py_logging.Handler} that remembers all logged records in a list.
    """

    def __init__(self) -> None:
        """
        Initialize this L{BufferedHandler}.
        """
        py_logging.Handler.__init__(self)
        self.records: List[LogRecord] = []

    def emit(self, record: LogRecord) -> None:
        """
        Remember the record.
        """
        self.records.append(record)