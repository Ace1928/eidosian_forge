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
def test_warningToFile(self) -> None:
    """
        L{twisted.python.log.showwarning} passes warnings with an explicit file
        target on to the underlying Python warning system.
        """
    message = 'another unique message'
    category = FakeWarning
    filename = 'warning-filename.py'
    lineno = 31
    output = StringIO()
    log.showwarning(message, category, filename, lineno, file=output)
    self.assertEqual(output.getvalue(), warnings.formatwarning(message, category, filename, lineno))
    line = 'hello world'
    output = StringIO()
    log.showwarning(message, category, filename, lineno, file=output, line=line)
    self.assertEqual(output.getvalue(), warnings.formatwarning(message, category, filename, lineno, line))