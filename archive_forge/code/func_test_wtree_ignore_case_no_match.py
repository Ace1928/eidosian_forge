import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_ignore_case_no_match(self):
    """(wtree) Match fails without --ignore-case.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt')
    out, err = self.run_bzr(['grep', 'LinE1', 'file0.txt'])
    self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '.inE1', 'file0.txt'])
    self.assertNotContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)