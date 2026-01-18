import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_unicode_backslashes(self):
    self.requireFeature(backslashdir_feature)
    self.build_unicode_tree()
    self._run_testset([[['ስ\\'], ['ስ/']], [['ስ\\ስ'], ['ስ/ስ']], [['ስ\\?'], ['ስ/ስ']], [['ስ\\*'], ['ስ/ስ']], [['?\\'], ['ስ/']], [['*\\'], ['ስ/']], [['?\\?'], ['ስ/ስ']], [['*\\*'], ['ስ/ስ']]])