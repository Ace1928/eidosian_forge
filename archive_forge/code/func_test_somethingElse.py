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
def test_somethingElse(self):
    """
        L{reporter.TestResult.addError} raises L{TypeError} if it is called with
        an error that is neither a L{sys.exc_info}-like three-tuple nor a
        L{Failure}.
        """
    with self.assertRaises(TypeError):
        self.result.addError(self, 'an error')
    with self.assertRaises(TypeError):
        self.result.addError(self, Exception('an error'))
    with self.assertRaises(TypeError):
        self.result.addError(self, (Exception, Exception('an error'), None, 'extra'))
    with self.assertRaises(TypeError):
        self.result.addError(self, (Exception, Exception('an error')))