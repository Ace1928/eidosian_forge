import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_binary_file_grep(self):
    """(versioned) Grep for pattern in binary file.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file.txt')
    self._mk_versioned_file('file0.bin')
    self._update_file('file0.bin', '\x00lineNN\x00\n')
    out, err = self.run_bzr(['grep', '-v', '-r', 'last:1', 'lineNN', 'file0.bin'])
    self.assertNotContainsRe(out, 'file0.bin', flags=TestGrep._reflags)
    self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    self.assertEqual(len(err.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-v', '-r', 'last:1', 'line.N', 'file0.bin'])
    self.assertNotContainsRe(out, 'file0.bin', flags=TestGrep._reflags)
    self.assertContainsRe(err, 'Binary file.*file0.bin.*skipped', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 0)
    self.assertEqual(len(err.splitlines()), 1)