from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
def test_startLoggingOverridesWarning(self) -> None:
    """
        startLogging() overrides global C{warnings.showwarning} such that
        warnings go to Twisted log observers.
        """
    self._startLoggingCleanup()
    newPublisher = NewLogPublisher()

    class SysModule:
        stdout = object()
        stderr = object()
    tempLogPublisher = LogPublisher(newPublisher, newPublisher, logBeginner=LogBeginner(newPublisher, StringIO(), SysModule, warnings))
    self.patch(log, 'theLogPublisher', tempLogPublisher)
    log._oldshowwarning = None
    fakeFile = StringIO()
    evt = {'pre-start': 'event'}
    received = []

    @implementer(ILogObserver)
    class PreStartObserver:

        def __call__(self, eventDict: log.EventDict) -> None:
            if 'pre-start' in eventDict.keys():
                received.append(eventDict)
    newPublisher(evt)
    newPublisher.addObserver(PreStartObserver())
    log.startLogging(fakeFile, setStdout=False)
    self.addCleanup(tempLogPublisher._stopLogging)
    self.assertEqual(received, [])
    warnings.warn('hello!')
    output = fakeFile.getvalue()
    self.assertIn('UserWarning: hello!', output)