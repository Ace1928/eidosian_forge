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
def test_writeLevel(self) -> None:
    """
        Log level is emitted properly.
        """
    f = self.observedFile()
    f.write('Hello\n')
    self.assertEqual(len(f.events), 1)
    self.assertEqual(f.events[0]['log_level'], LogLevel.info)
    f = self.observedFile(level=LogLevel.error)
    f.write('Hello\n')
    self.assertEqual(len(f.events), 1)
    self.assertEqual(f.events[0]['log_level'], LogLevel.error)