import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_file_within_dir(self):
    """(versioned) Search for pattern while in nested dir.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    os.chdir('dir0')
    out, err = self.run_bzr(['grep', '-r', 'last:1', 'line1'])
    self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '.i.e1'])
    self.assertContainsRe(out, '^file0.txt~.:line1', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)