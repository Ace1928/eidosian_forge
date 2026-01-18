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
def test_publisherReportsBrokenObserversPrivately(self) -> None:
    """
        Log publisher does not use the global L{log.err} when reporting broken
        observers.
        """
    errors = []

    def logError(eventDict: log.EventDict) -> None:
        if eventDict.get('isError'):
            errors.append(eventDict['failure'].value)

    def fail(eventDict: log.EventDict) -> None:
        raise RuntimeError('test_publisherLocalyReportsBrokenObservers')
    publisher = log.LogPublisher()
    publisher.addObserver(logError)
    publisher.addObserver(fail)
    publisher.msg('Hello!')
    self.assertEqual(set(publisher.observers), {logError, fail})
    self.assertEqual(len(errors), 1)
    self.assertIsInstance(errors[0], RuntimeError)