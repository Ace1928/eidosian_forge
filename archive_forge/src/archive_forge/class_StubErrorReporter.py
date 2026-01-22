from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
class StubErrorReporter:
    """
    A subset of L{twisted.trial.itrial.IReporter} which records L{addError}
    calls.

    @ivar errors: List of two-tuples of (test, error) which were passed to
        L{addError}.
    """

    def __init__(self) -> None:
        self.errors: list[tuple[object, Failure]] = []

    def addError(self, test: object, error: Failure) -> None:
        """
        Record parameters in C{self.errors}.
        """
        self.errors.append((test, error))