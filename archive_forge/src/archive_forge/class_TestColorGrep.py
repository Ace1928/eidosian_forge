import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
class TestColorGrep(GrepTestBase):
    """Tests for the --color option."""
    _rev_sep = color_string('~', fg=FG.BOLD_YELLOW)
    _sep = color_string(':', fg=FG.BOLD_CYAN)

    def test_color_option(self):
        """Ensure options for color are valid.
        """
        out, err = self.run_bzr(['grep', '--color', 'foo', 'bar'], 3)
        self.assertEqual(out, '')
        self.assertContainsRe(err, 'Valid values for --color are', flags=TestGrep._reflags)

    def test_ver_matching_files(self):
        """(versioned) Search for matches or no matches only"""
        tree = self.make_branch_and_tree('.')
        contents = ['d/', 'd/aaa', 'bbb']
        self.build_tree(contents)
        tree.add(contents)
        tree.commit('Initial commit')
        streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-with-matches', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', self._rev_sep, '1', '\n']), ''))
        streams = self.run_bzr(['grep', '--color', 'always', '-r', '1', '--files-without-match', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', self._rev_sep, '1', '\n']), ''))

    def test_wtree_matching_files(self):
        """(wtree) Search for matches or no matches only"""
        tree = self.make_branch_and_tree('.')
        contents = ['d/', 'd/aaa', 'bbb']
        self.build_tree(contents)
        tree.add(contents)
        tree.commit('Initial commit')
        streams = self.run_bzr(['grep', '--color', 'always', '--files-with-matches', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'd/aaa', FG.NONE, '\n']), ''))
        streams = self.run_bzr(['grep', '--color', 'always', '--files-without-match', 'aaa'])
        self.assertEqual(streams, (''.join([FG.MAGENTA, 'bbb', FG.NONE, '\n']), ''))

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

    def test_wtree_basic_file(self):
        """(wtree) Search for pattern in specfic file.
        """
        wd = 'foobar0'
        self.make_branch_and_tree(wd)
        os.chdir(wd)
        lp = 'foo is foobar'
        self._mk_versioned_file('file0.txt', line_prefix=lp, total_lines=1)
        foo = color_string('foo', fg=FG.BOLD_RED)
        res = FG.MAGENTA + 'file0.txt' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        nres = FG.MAGENTA + 'file0.txt' + self._sep + '1' + self._sep + foo + ' is ' + foo + 'bar1' + '\n'
        out, err = self.run_bzr(['grep', '--color', 'always', 'foo'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-i', '--color', 'always', 'FOO'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '--color', 'always', 'f.o'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-i', '--color', 'always', 'F.O'])
        self.assertEqual(out, res)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-n', '--color', 'always', 'foo'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-n', '-i', '--color', 'always', 'FOO'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-n', '--color', 'always', 'f.o'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)
        out, err = self.run_bzr(['grep', '-n', '-i', '--color', 'always', 'F.O'])
        self.assertEqual(out, nres)
        self.assertEqual(len(out.splitlines()), 1)