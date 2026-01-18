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
def test_singleString(self) -> None:
    """
        Test simple output, and default log level.
        """
    self.lp.msg('Hello, world.')
    self.assertIn(b'Hello, world.', self.out.getvalue())
    self.assertIn(b'INFO', self.out.getvalue())