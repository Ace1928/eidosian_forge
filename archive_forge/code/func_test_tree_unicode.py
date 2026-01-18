import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_tree_unicode(self):
    """Checks behaviour with non-ascii filenames"""
    self.build_unicode_tree()
    self._run_testset([[['ሴ'], ['ሴ']], [['ስ'], ['ስ']], [['ስ/'], ['ስ/']], [['ስ/ስ'], ['ስ/ስ']], [['?'], ['ሴ', 'ስ']], [['*'], ['ሴ', 'ሴሴ', 'ስ']], [['ሴ*'], ['ሴ', 'ሴሴ']], [['ስ/?'], ['ስ/ስ']], [['ስ/*'], ['ስ/ስ']], [['?/'], ['ስ/']], [['*/'], ['ስ/']], [['?/?'], ['ስ/ስ']], [['*/*'], ['ስ/ስ']]])