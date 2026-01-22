import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
class LoggingReporter(reporter.Reporter):
    """
    Simple reporter that stores the last test that was passed to it.
    """

    def __init__(self, *args, **kwargs):
        reporter.Reporter.__init__(self, *args, **kwargs)
        self.test = None

    def addError(self, test, error):
        self.test = test

    def addExpectedFailure(self, test, failure, todo=None):
        self.test = test

    def addFailure(self, test, failure):
        self.test = test

    def addSkip(self, test, skip):
        self.test = test

    def addUnexpectedSuccess(self, test, todo=None):
        self.test = test

    def startTest(self, test):
        self.test = test

    def stopTest(self, test):
        self.test = test