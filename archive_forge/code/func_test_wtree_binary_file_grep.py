import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_binary_file_grep(self):
    """(wtree) Grep for pattern in binary file.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.bin')
    self._update_file('file0.bin', '\x00lineNN\x00\n')
    out, err = self.run_bzr(['grep', '-v', 'lineNN', 'file0.bin'])
    self.assertNotContainsRe(out, 'file0.bin:line1', flags=TestGrep._reflags)
    self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
    out, err = self.run_bzr(['grep', 'lineNN', 'file0.bin'])
    self.assertNotContainsRe(out, 'file0.bin:line1', flags=TestGrep._reflags)
    self.assertNotContainsRe(err, 'Binary file', flags=TestGrep._reflags)