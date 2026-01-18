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
def test_printResults(self):
    """
        L{Reporter._printResults} uses the results list and formatter callable
        passed to it to produce groups of results to write to its output
        stream.
        """

    def formatter(n):
        return str(n) + '\n'
    first = sample.FooTest('test_foo')
    second = sample.FooTest('test_bar')
    third = sample.PyunitTest('test_foo')
    self.result._printResults('FOO', [(first, 1), (second, 1), (third, 2)], formatter)
    self.assertEqual(self.stream.getvalue(), '%(double separator)s\nFOO\n1\n\n%(first)s\n%(second)s\n%(double separator)s\nFOO\n2\n\n%(third)s\n' % {'double separator': self.result._doubleSeparator, 'first': first.id(), 'second': second.id(), 'third': third.id()})