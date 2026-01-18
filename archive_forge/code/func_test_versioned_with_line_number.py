import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_with_line_number(self):
    """(versioned) Search for pattern with --line-number.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--line-number', 'li.e3', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt~.:3:line3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--line-number', 'line3', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt~.:3:line3', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '-n', 'line1', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt~.:1:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '-n', 'line[0-9]', 'file0.txt'])
    self.assertContainsRe(out, 'file0.txt:3:line3', flags=TestGrep._reflags)