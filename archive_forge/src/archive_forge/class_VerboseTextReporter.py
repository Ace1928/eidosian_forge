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
class VerboseTextReporter(Reporter):
    """
    A verbose reporter that prints the name of each test as it is running.

    Each line is printed with the name of the test, followed by the result of
    that test.
    """

    def startTest(self, tm):
        self._write('%s ... ', tm.id())
        super().startTest(tm)

    def addSuccess(self, test):
        super().addSuccess(test)
        self._write('[OK]')

    def addError(self, *args):
        super().addError(*args)
        self._write('[ERROR]')

    def addFailure(self, *args):
        super().addFailure(*args)
        self._write('[FAILURE]')

    def addSkip(self, *args):
        super().addSkip(*args)
        self._write('[SKIPPED]')

    def addExpectedFailure(self, *args):
        super().addExpectedFailure(*args)
        self._write('[TODO]')

    def addUnexpectedSuccess(self, *args):
        super().addUnexpectedSuccess(*args)
        self._write('[SUCCESS!?!]')

    def stopTest(self, test):
        super().stopTest(test)
        self._write('\n')