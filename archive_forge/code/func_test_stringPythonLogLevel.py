import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
def test_stringPythonLogLevel(self) -> None:
    """
        If a stdlib log level was provided as a string (eg. C{"WARNING"}) in
        the legacy "logLevel" key, it does not get converted to a number.
        The documentation suggested that numerical values should be used but
        this was not a requirement.
        """
    event = self.forwardAndVerify(dict(logLevel='WARNING'))
    self.assertEqual(event['logLevel'], 'WARNING')