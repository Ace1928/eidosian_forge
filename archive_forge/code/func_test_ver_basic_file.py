import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def test_ver_basic_file(self):
    """(versioned) Search for pattern in specfic file.
        """
    wd = 'foobar0'
    self.make_branch_and_tree(wd)
    os.chdir(wd)
    lp = 'foo is foobar'
    self._mk_versioned_file('file0.txt', line_prefix=lp, total_lines=1)
    foo = color_string('foo', fg=FG.BOLD_RED)
    res = FG.MAGENTA + 'file0.txt' + self._rev_sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
    txt_res = 'file0.txt~1:foo is foobar1\n'
    nres = FG.MAGENTA + 'file0.txt' + self._rev_sep + '1' + self._sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
    out, err = self.run_bzr(['grep', '--color', 'always', '-r', '1', 'foo'])
    self.assertEqual(out, res)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '--color', 'auto', '-r', '1', 'foo'])
    self.assertEqual(out, txt_res)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-i', '--color', 'always', '-r', '1', 'FOO'])
    self.assertEqual(out, res)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '--color', 'always', '-r', '1', 'f.o'])
    self.assertEqual(out, res)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-i', '--color', 'always', '-r', '1', 'F.O'])
    self.assertEqual(out, res)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-n', '--color', 'always', '-r', '1', 'foo'])
    self.assertEqual(out, nres)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-n', '-i', '--color', 'always', '-r', '1', 'FOO'])
    self.assertEqual(out, nres)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-n', '--color', 'always', '-r', '1', 'f.o'])
    self.assertEqual(out, nres)
    self.assertEqual(len(out.splitlines()), 1)
    out, err = self.run_bzr(['grep', '-n', '-i', '--color', 'always', '-r', '1', 'F.O'])
    self.assertEqual(out, nres)
    self.assertEqual(len(out.splitlines()), 1)