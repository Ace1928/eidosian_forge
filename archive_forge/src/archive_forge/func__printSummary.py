from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def _printSummary(self):
    """
        Print a line summarising the test results to the stream, and color the
        status result.
        """
    summary = self._getSummary()
    if self.wasSuccessful():
        status = 'PASSED'
        color = self.SUCCESS
    else:
        status = 'FAILED'
        color = self.FAILURE
    self._colorizer.write(status, color)
    self._write('%s\n', summary)