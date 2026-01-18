from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
def test_testmoduleOnNonexistentFile(self) -> None:
    """
        Check that --testmodule displays a meaningful error message when
        passed a non-existent filename.
        """
    buffy = StringIO()
    stderr, sys.stderr = (sys.stderr, buffy)
    filename = 'test_thisbetternoteverexist.py'
    try:
        self.config.opt_testmodule(filename)
        self.assertEqual(0, len(self.config['tests']))
        self.assertEqual(f"File {filename!r} doesn't exist\n", buffy.getvalue())
    finally:
        sys.stderr = stderr