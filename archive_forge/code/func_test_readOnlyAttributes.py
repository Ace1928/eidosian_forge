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
def test_readOnlyAttributes(self) -> None:
    """
        Some L{LoggingFile} attributes are read-only.
        """
    f = LoggingFile(self.logger)
    self.assertRaises(AttributeError, setattr, f, 'closed', True)
    self.assertRaises(AttributeError, setattr, f, 'encoding', 'utf-8')
    self.assertRaises(AttributeError, setattr, f, 'mode', 'r')
    self.assertRaises(AttributeError, setattr, f, 'newlines', ['\n'])
    self.assertRaises(AttributeError, setattr, f, 'name', 'foo')