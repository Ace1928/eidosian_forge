import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_with_line_number(self):
    """(wtree) Search for pattern with --line-number.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt')
    out, err = self.run_bzr(['grep', '--line-number', 'line3', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-n', 'line1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:1:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-n', '[hjkl]ine1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:1:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-n', 'line[0-9]', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)