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
@implementer(ILogObserver)
class PreStartObserver:

    def __call__(self, eventDict: log.EventDict) -> None:
        if 'pre-start' in eventDict.keys():
            received.append(eventDict)