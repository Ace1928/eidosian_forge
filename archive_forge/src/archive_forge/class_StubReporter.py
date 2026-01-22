from __future__ import annotations
import sys
import traceback
import unittest as pyunit
from unittest import skipIf
from zope.interface import implementer
from twisted.python.failure import Failure
from twisted.trial.itrial import IReporter, ITestCase
from twisted.trial.test import pyunitcases
from twisted.trial.unittest import PyUnitResultAdapter, SynchronousTestCase
@implementer(IReporter)
class StubReporter:
    """
            A reporter which records data about calls made to it.

            @ivar errors: Errors passed to L{addError}.
            @ivar failures: Failures passed to L{addFailure}.
            """

    def __init__(self) -> None:
        self.errors: list[Failure] = []
        self.failures: list[None] = []

    def startTest(self, test: object) -> None:
        """
                Do nothing.
                """

    def stopTest(self, test: object) -> None:
        """
                Do nothing.
                """

    def addError(self, test: object, error: Failure) -> None:
        """
                Record the error.
                """
        self.errors.append(error)