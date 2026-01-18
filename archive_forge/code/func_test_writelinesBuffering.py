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
def test_writelinesBuffering(self) -> None:
    """
        C{writelines} does not add newlines.
        """
    f = self.observedFile()
    f.writelines(('Hello', ', ', ''))
    self.assertEqual(f.messages, [])
    f.writelines(('world!\n',))
    self.assertEqual(f.messages, ['Hello, world!'])
    f.writelines(("It's nice to meet you.\n\n", 'Indeed.'))
    self.assertEqual(f.messages, ['Hello, world!', "It's nice to meet you.", ''])