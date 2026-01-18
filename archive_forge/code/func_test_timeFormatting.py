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
def test_timeFormatting(self) -> None:
    """
        Test the method of L{FileLogObserver} which turns a timestamp into a
        human-readable string.
        """
    when = calendar.timegm((2001, 2, 3, 4, 5, 6, 7, 8, 0))
    self.flo.getTimezoneOffset = lambda when: 18000
    self.assertEqual(self.flo.formatTime(when), '2001-02-02 23:05:06-0500')
    self.flo.getTimezoneOffset = lambda when: -3600
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 05:05:06+0100')
    self.flo.getTimezoneOffset = lambda when: -39600
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 15:05:06+1100')
    self.flo.getTimezoneOffset = lambda when: 5400
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 02:35:06-0130')
    self.flo.getTimezoneOffset = lambda when: -5400
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 05:35:06+0130')
    self.flo.getTimezoneOffset = lambda when: 1800
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 03:35:06-0030')
    self.flo.getTimezoneOffset = lambda when: -1800
    self.assertEqual(self.flo.formatTime(when), '2001-02-03 04:35:06+0030')
    self.flo.timeFormat = '%Y %m'
    self.assertEqual(self.flo.formatTime(when), '2001 02')