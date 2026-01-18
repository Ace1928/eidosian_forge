import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_versioned_include_from_outside_dir(self):
    """(versioned) Ensure --include is respected during recursive search.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.aa')
    self._mk_versioned_dir('dir1')
    self._mk_versioned_file('dir1/file1.bb')
    self._mk_versioned_dir('dir2')
    self._mk_versioned_file('dir2/file2.cc')
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'l..e1'])
    self.assertContainsRe(out, '^dir0/file0.aa~.:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.bb~.:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.aa~.:line10$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.bb~.:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)
    out, err = self.run_bzr(['grep', '-r', 'last:1', '--include', '*.aa', '--include', '*.bb', 'line1'])
    self.assertContainsRe(out, '^dir0/file0.aa~.:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.bb~.:line1$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.aa~.:line10$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir1/file1.bb~.:line10$', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, 'file1.cc', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 4)