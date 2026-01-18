import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_files_with_matches(self):
    """(wtree) Ensure --files-with-matches, -l works
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt', total_lines=2)
    self._mk_versioned_file('file1.txt', total_lines=2)
    self._mk_versioned_dir('dir0')
    self._mk_versioned_file('dir0/file00.txt', total_lines=2)
    self._mk_versioned_file('dir0/file01.txt', total_lines=2)
    self._update_file('file0.txt', 'HELLO\n', checkin=False)
    self._update_file('dir0/file00.txt', 'HELLO\n', checkin=False)
    out, err = self.run_bzr(['grep', '--files-with-matches', 'HELLO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '--files-with-matches', 'HE.LO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-l', 'HELLO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-l', 'HE.LO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 2)
    out, err = self.run_bzr(['grep', '-l', 'HELLO', 'dir0', 'file1.txt'])
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-l', '.ELLO', 'dir0', 'file1.txt'])
    self.assertContainsRe(out, '^dir0/file00.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-l', 'HELLO', 'file0.txt'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-l', '.ELLO', 'file0.txt'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '--no-recursive', '-l', 'HELLO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '--no-recursive', '-l', '.ELLO'])
    self.assertContainsRe(out, '^file0.txt$', flags=TestGrep._reflags)
    self.assertEqual(len(out.splitlines()), 1)