import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_include_file_within_dir(self):
    """(versioned) Ensure --include is respected with file within dir.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._mk_versioned_file('dir0/file1.aa')
    self._update_file('dir0/file1.aa', 'hello\n')
    self._update_file('dir0/file0.txt', 'hello\n')
    os.chdir('dir0')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', 'line1'])
    self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-r', 'last:2..last:1', '--include', '*.aa', 'line1'])
    self.assertContainsRe(out, '^file1.aa~4:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~4:line10$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', 'lin.1'])
    self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-r', 'last:3..last:1', '--include', '*.aa', 'lin.1'])
    self.assertContainsRe(out, '^file1.aa~3:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~4:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~3:line10$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~4:line10$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^file1.aa~5:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file0.txt', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 6)