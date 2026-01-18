import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_from_root_pass(self):
    """(wtree) Match pass with --from-root.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt')
    self._mk_versioned_dir('dir0')
    os.chdir('dir0')
    out, err = self.run_bzr(['grep', '--from-root', 'lin.1'])
    self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', '--from-root', 'line1'])
    self.assertContainsRe(out, 'file0.txt:line1', flags=TestGrep._reflags)