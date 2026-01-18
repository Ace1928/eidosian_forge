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
def test_strangeEventDict(self) -> None:
    """
        Verify that an event dictionary which is not an error and has an empty
        message isn't recorded.
        """
    self.lp.msg(message='', isError=False)
    self.assertEqual(self.out.getvalue(), b'')