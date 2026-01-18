import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_wtree_null_option(self):
    """(wtree) --null option should use NUL instead of newline.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    self._mk_versioned_file('file0.txt', total_lines=3)
    out, err = self.run_bzr(['grep', '--null', 'line[1-3]'])
    self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-Z', 'line[1-3]'])
    self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-Z', 'line'])
    self.assertEqual(out, 'file0.txt:line1\x00file0.txt:line2\x00file0.txt:line3\x00')
    self.assertEqual(len(out.splitlines()), 1)