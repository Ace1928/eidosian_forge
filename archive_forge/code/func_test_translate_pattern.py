import os
import re
import unittest
from distutils import debug
from distutils.log import WARN
from distutils.errors import DistutilsTemplateError
from distutils.filelist import glob_to_re, translate_pattern, FileList
from distutils import filelist
from test.support import os_helper
from test.support import captured_stdout
from distutils.tests import support
def test_translate_pattern(self):
    self.assertTrue(hasattr(translate_pattern('a', anchor=True, is_regex=False), 'search'))
    regex = re.compile('a')
    self.assertEqual(translate_pattern(regex, anchor=True, is_regex=True), regex)
    self.assertTrue(hasattr(translate_pattern('a', anchor=True, is_regex=True), 'search'))
    self.assertTrue(translate_pattern('*.py', anchor=True, is_regex=False).search('filelist.py'))