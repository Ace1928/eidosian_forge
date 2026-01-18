import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def test_unicode_dir(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    os.mkdir('ሴ')
    win32utils.set_file_attr_hidden('ሴ')