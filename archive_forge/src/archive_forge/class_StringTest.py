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
class StringTest(unittest.SynchronousTestCase):

    def stringComparison(self, expect, output):
        output = list(filter(None, output))
        self.assertTrue(len(expect) <= len(output), 'Must have more observed than expectedlines %d < %d' % (len(output), len(expect)))
        REGEX_PATTERN_TYPE = type(re.compile(''))
        for line_number, (exp, out) in enumerate(zip(expect, output)):
            if exp is None:
                continue
            elif isinstance(exp, str):
                self.assertSubstring(exp, out, 'Line %d: %r not in %r' % (line_number, exp, out))
            elif isinstance(exp, REGEX_PATTERN_TYPE):
                self.assertTrue(exp.match(out), 'Line %d: %r did not match string %r' % (line_number, exp.pattern, out))
            else:
                raise TypeError(f"don't know what to do with object {exp!r}")