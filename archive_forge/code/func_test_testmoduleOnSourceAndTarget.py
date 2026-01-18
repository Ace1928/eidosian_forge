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
def test_testmoduleOnSourceAndTarget(self) -> None:
    """
        If --testmodule is specified twice, once for module A and once for
        a module which refers to module A, then make sure module A is only
        added once.
        """
    self.config.opt_testmodule(sibpath('moduletest.py'))
    self.config.opt_testmodule(sibpath('test_log.py'))
    self.assertSuitesEqual(trial._getSuite(self.config), ['twisted.trial.test.test_log'])