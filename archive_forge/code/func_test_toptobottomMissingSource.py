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
def test_toptobottomMissingSource(self) -> None:
    """
        --order=toptobottom detects the source line of methods from modules
        whose source file is missing.
        """
    tempdir = self.mktemp()
    package = FilePath(tempdir).child('twisted_toptobottom_temp')
    package.makedirs()
    package.child('__init__.py').setContent(b'')
    package.child('test_missing.py').setContent(textwrap.dedent('\n        from twisted.trial.unittest import TestCase\n        class TestMissing(TestCase):\n            def test_second(self) -> None: pass\n            def test_third(self) -> None: pass\n            def test_fourth(self) -> None: pass\n            def test_first(self) -> None: pass\n        ').encode('utf8'))
    pathEntry = package.parent().path
    sys.path.insert(0, pathEntry)
    self.addCleanup(sys.path.remove, pathEntry)
    from twisted_toptobottom_temp import test_missing
    self.addCleanup(sys.modules.pop, 'twisted_toptobottom_temp')
    self.addCleanup(sys.modules.pop, test_missing.__name__)
    package.child('test_missing.py').remove()
    self.config.parseOptions(['--order', 'toptobottom', 'twisted.trial.test.ordertests'])
    loader = trial._getLoader(self.config)
    suite = loader.loadModule(test_missing)
    self.assertEqual(testNames(suite), ['twisted_toptobottom_temp.test_missing.TestMissing.test_second', 'twisted_toptobottom_temp.test_missing.TestMissing.test_third', 'twisted_toptobottom_temp.test_missing.TestMissing.test_fourth', 'twisted_toptobottom_temp.test_missing.TestMissing.test_first'])