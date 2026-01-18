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
def test_whyDefault(self) -> None:
    """
        The C{"why"} value, when unspecified, defaults to C{"Unhandled Error"}.
        """
    try:
        raise RuntimeError()
    except BaseException:
        eventDict = dict(message=(), isError=1, failure=failure.Failure())
        text = log.textFromEventDict(eventDict)
        assert text is not None
        self.assertTrue(text.startswith('Unhandled Error\n'))