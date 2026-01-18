import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_quoted_globs(self):
    self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
    self.assertCommandLine(['a/*.c'], '"a/*.c"')
    self.assertCommandLine(["'a/*.c'"], "'a/*.c'")
    self.assertCommandLine(['a/*.c'], "'a/*.c'", single_quotes_allowed=True)