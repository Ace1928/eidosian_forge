import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_single_quote_support(self):
    self.assertCommandLine(['add', "let's-do-it.txt"], "add let's-do-it.txt", ['add', "let's-do-it.txt"])
    self.expectFailure('Using single quotes breaks trimming from argv', self.assertCommandLine, ['add', 'lets do it.txt'], "add 'lets do it.txt'", ['add', "'lets", 'do', "it.txt'"], single_quotes_allowed=True)