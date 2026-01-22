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
class BrokenStream:
    """
    Stream-ish object that raises a signal interrupt error. We use this to make
    sure that Trial still manages to write what it needs to write.
    """
    written = False
    flushed = False

    def __init__(self, fObj):
        self.fObj = fObj

    def write(self, s):
        if self.written:
            return self.fObj.write(s)
        self.written = True
        raise OSError(errno.EINTR, 'Interrupted write')

    def flush(self):
        if self.flushed:
            return self.fObj.flush()
        self.flushed = True
        raise OSError(errno.EINTR, 'Interrupted flush')