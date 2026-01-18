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
def test_writeBytesDecoded(self) -> None:
    """
        Bytes are decoded to text.
        """
    f = self.observedFile(encoding='utf-8')
    f.write(b'Hello, Mr. S\xc3\xa1nchez\n')
    self.assertEqual(f.messages, ['Hello, Mr. Sánchez'])