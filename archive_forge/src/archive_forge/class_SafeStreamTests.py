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
class SafeStreamTests(unittest.SynchronousTestCase):

    def test_safe(self):
        """
        Test that L{reporter.SafeStream} successfully write to its original
        stream even if an interrupt happens during the write.
        """
        stream = StringIO()
        broken = BrokenStream(stream)
        safe = reporter.SafeStream(broken)
        safe.write('Hello')
        self.assertEqual(stream.getvalue(), 'Hello')