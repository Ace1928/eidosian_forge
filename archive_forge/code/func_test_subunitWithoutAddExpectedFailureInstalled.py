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
def test_subunitWithoutAddExpectedFailureInstalled(self):
    """
        Some versions of subunit don't have "addExpectedFailure". For these
        versions, we report expected failures as successes.
        """
    self.removeMethod(reporter.TestProtocolClient, 'addExpectedFailure')
    try:
        1 / 0
    except ZeroDivisionError:
        self.result.addExpectedFailure(self.test, sys.exc_info(), 'todo')
    expectedFailureOutput = self.stream.getvalue()
    self.stream.truncate(0)
    self.stream.seek(0)
    self.result.addSuccess(self.test)
    successOutput = self.stream.getvalue()
    self.assertEqual(successOutput, expectedFailureOutput)