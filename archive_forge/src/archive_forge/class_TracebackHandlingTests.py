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
class TracebackHandlingTests(unittest.SynchronousTestCase):

    def getErrorFrames(self, test):
        """
        Run the given C{test}, make sure it fails and return the trimmed
        frames.

        @param test: The test case to run.

        @return: The C{list} of frames trimmed.
        """
        stream = StringIO()
        result = reporter.Reporter(stream)
        test.run(result)
        bads = result.failures + result.errors
        self.assertEqual(len(bads), 1)
        self.assertEqual(bads[0][0], test)
        return result._trimFrames(bads[0][1].frames)

    def checkFrames(self, observedFrames, expectedFrames):
        for observed, expected in zip(observedFrames, expectedFrames):
            self.assertEqual(observed[0], expected[0])
            observedSegs = os.path.splitext(observed[1])[0].split(os.sep)
            expectedSegs = expected[1].split('/')
            self.assertEqual(observedSegs[-len(expectedSegs):], expectedSegs)
        self.assertEqual(len(observedFrames), len(expectedFrames))

    def test_basic(self):
        test = erroneous.TestRegularFail('test_fail')
        frames = self.getErrorFrames(test)
        self.checkFrames(frames, [('test_fail', 'twisted/trial/test/erroneous')])

    def test_subroutine(self):
        test = erroneous.TestRegularFail('test_subfail')
        frames = self.getErrorFrames(test)
        self.checkFrames(frames, [('test_subfail', 'twisted/trial/test/erroneous'), ('subroutine', 'twisted/trial/test/erroneous')])

    def test_deferred(self):
        """
        C{_trimFrames} removes traces of C{_runCallbacks} when getting an error
        in a callback returned by a C{TestCase} based test.
        """
        test = erroneous.TestAsynchronousFail('test_fail')
        frames = self.getErrorFrames(test)
        self.checkFrames(frames, [('cb', 'twisted/internet/task')])

    def test_noFrames(self):
        result = reporter.Reporter(None)
        self.assertEqual([], result._trimFrames([]))

    def test_oneFrame(self):
        result = reporter.Reporter(None)
        self.assertEqual(['fake frame'], result._trimFrames(['fake frame']))

    def test_exception(self):
        """
        C{_trimFrames} removes traces of C{runWithWarningsSuppressed} from
        C{utils} when a synchronous exception happens in a C{TestCase}
        based test.
        """
        test = erroneous.TestAsynchronousFail('test_exception')
        frames = self.getErrorFrames(test)
        self.checkFrames(frames, [('test_exception', 'twisted/trial/test/erroneous')])