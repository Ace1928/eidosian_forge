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
def _startLoggingCleanup(self) -> None:
    """
        Cleanup after a startLogging() call that mutates the hell out of some
        global state.
        """
    self.addCleanup(log.theLogPublisher._stopLogging)
    self.addCleanup(setattr, sys, 'stdout', sys.stdout)
    self.addCleanup(setattr, sys, 'stderr', sys.stderr)