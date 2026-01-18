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
def test_supportedSetupTerm(self):
    """
        L{reporter._AnsiColorizer.supported} returns C{True} if
        C{curses.tigetnum} returns more than 2 supported colors. It only tries
        to call C{curses.setupterm} if C{curses.tigetnum} previously failed
        with a C{curses.error}.
        """

    class fakecurses:
        error = RuntimeError
        setUp = 0

        def setupterm(self):
            self.setUp += 1

        def tigetnum(self, value):
            if self.setUp:
                return 3
            else:
                raise self.error()
    sys.modules['curses'] = fakecurses()
    self.assertTrue(reporter._AnsiColorizer.supported(FakeStream()))
    self.assertTrue(reporter._AnsiColorizer.supported(FakeStream()))
    self.assertEqual(sys.modules['curses'].setUp, 1)