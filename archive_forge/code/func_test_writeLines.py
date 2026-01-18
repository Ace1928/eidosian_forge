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
def test_writeLines(self) -> None:
    """
        Writing lines to a StdioOnnaStick results in Twisted log messages.
        """
    stdio = log.StdioOnnaStick()
    stdio.writelines(['log 1', 'log 2'])
    self.assertEqual(self.getLogMessages(), ['log 1', 'log 2'])