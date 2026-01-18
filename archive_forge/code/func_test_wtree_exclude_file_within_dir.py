import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_exclude_file_within_dir(self):
    """(wtree) Ensure --exclude is respected with file within dir.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._mk_versioned_file('dir0/file1.aa')
    os.chdir('dir0')
    out, err = self.run_bzr(['grep', '--exclude', '*.txt', 'li.e1'])
    self.assertContainsRe(out, '^file1.aa:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '--exclude', '*.txt', 'line1'])
    self.assertContainsRe(out, '^file1.aa:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)