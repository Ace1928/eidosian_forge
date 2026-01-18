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
def test_printToStderrSetsIsError(self) -> None:
    """
        startLogging()'s overridden sys.stderr should consider everything
        written to it an error.
        """
    self._startLoggingCleanup()
    fakeFile = StringIO()
    log.startLogging(fakeFile)

    def observe(event: log.EventDict) -> None:
        observed.append(event)
    observed: list[log.EventDict] = []
    log.addObserver(observe)
    print('Hello, world.', file=sys.stderr)
    self.assertEqual(observed[0]['isError'], 1)