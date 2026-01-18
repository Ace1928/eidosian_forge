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
def test_warningEmittedForNewTest(self):
    """
        A warning emitted again after a new test has started is written to the
        stream again.
        """
    test = self.__class__('test_warningEmittedForNewTest')
    self.result.startTest(test)
    self.stream.seek(0)
    self.stream.truncate()
    self.test_warning()
    self.stream.seek(0)
    self.stream.truncate()
    self.result.stopTest(test)
    self.result.startTest(test)
    self.stream.seek(0)
    self.stream.truncate()
    self.test_warning()