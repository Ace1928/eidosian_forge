import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
class TestGrepDiff(tests.TestCaseWithTransport):

    def make_example_branch(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([('hello', b'foo\n'), ('goodbye', b'baz\n')])
        tree.add(['hello'])
        tree.commit('setup')
        tree.add(['goodbye'])
        tree.commit('setup')
        return tree

    def test_grep_diff_basic(self):
        """grep -p basic test."""
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        tree.commit('updated hello')
        out, err = self.run_bzr(['grep', '-p', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n=== revno:1 ===\n  === added file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n")

    def test_grep_diff_revision(self):
        """grep -p specific revision."""
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        tree.commit('updated hello')
        out, err = self.run_bzr(['grep', '-p', '-r', '3', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n")

    def test_grep_diff_revision_range(self):
        """grep -p revision range."""
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!1\n')])
        tree.commit('rev3')
        self.build_tree_contents([('blah', b'hello world!2\n')])
        tree.add('blah')
        tree.commit('rev4')
        with open('hello', 'a') as f:
            f.write('hello world!3\n')
        tree.commit('rev5')
        out, err = self.run_bzr(['grep', '-p', '-r', '2..5', 'hello'])
        self.assertEqual(err, '')
        self.assertEqualDiff(subst_dates(out), "=== revno:5 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!3\n=== revno:4 ===\n  === added file 'blah'\n    +hello world!2\n=== revno:3 ===\n  === modified file 'hello'\n    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!1\n")

    def test_grep_diff_color(self):
        """grep -p color test."""
        tree = self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        tree.commit('updated hello')
        out, err = self.run_bzr(['grep', '--diff', '-r', '3', '--color', 'always', 'hello'])
        self.assertEqual(err, '')
        revno = color_string('=== revno:3 ===', fg=FG.BOLD_BLUE) + '\n'
        filename = color_string("  === modified file 'hello'", fg=FG.BOLD_MAGENTA) + '\n'
        redhello = color_string('hello', fg=FG.BOLD_RED)
        diffstr = '    --- hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +++ hello\tYYYY-MM-DD HH:MM:SS +ZZZZ\n    +hello world!\n'
        diffstr = diffstr.replace('hello', redhello)
        self.assertEqualDiff(subst_dates(out), revno + filename + diffstr)

    def test_grep_norevs(self):
        """grep -p with zero revisions."""
        out, err = self.run_bzr(['init'])
        out, err = self.run_bzr(['grep', '--diff', 'foo'], 3)
        self.assertEqual(out, '')
        self.assertContainsRe(err, 'ERROR:.*revision.* does not exist in branch')