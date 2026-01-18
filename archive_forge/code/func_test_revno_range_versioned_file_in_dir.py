import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_revno_range_versioned_file_in_dir(self):
    """Grep rev-range for pattern for file withing a dir.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file0.txt')
    self._update_file('dir0/file0.txt', 'v3 text\n')
    self._update_file('dir0/file0.txt', 'v4 text\n')
    self._update_file('dir0/file0.txt', 'v5 text\n')
    self._update_file('dir0/file0.txt', 'v6 text\n')
    out, err = self.run_bzr(['grep', '-r', '2..5', 'v3'])
    self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 3)
    out, err = self.run_bzr(['grep', '-r', '2..5', '[tuv]3'])
    self.assertContainsRe(out, '^dir0/file0.txt~3:v3', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.txt~4:v3', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file0.txt~5:v3', flags=TestGrep._reflags)
    self.assertNotContainsRe(out, '^dir0/file0.txt~6:v3', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 3)