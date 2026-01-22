import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
class Test_CommandLineToArgv(tests.TestCaseInTempDir):

    def assertCommandLine(self, expected, line, argv=None, single_quotes_allowed=False):
        if argv is None:
            argv = [line]
        argv = win32utils._command_line_to_argv(line, argv, single_quotes_allowed=single_quotes_allowed)
        self.assertEqual(expected, sorted(argv))

    def test_glob_paths(self):
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine(['a/b.c', 'a/c.c'], 'a/*.c')
        self.build_tree(['b/', 'b/b.c', 'b/d.c', 'b/d.h'])
        self.assertCommandLine(['a/b.c', 'b/b.c'], '*/b.c')
        self.assertCommandLine(['a/b.c', 'a/c.c', 'b/b.c', 'b/d.c'], '*/*.c')
        self.assertCommandLine(['*/*.qqq'], '*/*.qqq')

    def test_quoted_globs(self):
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine(['a/*.c'], '"a/*.c"')
        self.assertCommandLine(["'a/*.c'"], "'a/*.c'")
        self.assertCommandLine(['a/*.c'], "'a/*.c'", single_quotes_allowed=True)

    def test_slashes_changed(self):
        self.assertCommandLine(['a\\*.c'], '"a\\*.c"')
        self.assertCommandLine(['a\\*.c'], "'a\\*.c'", single_quotes_allowed=True)
        self.assertCommandLine(['a/*.c'], 'a\\*.c')
        self.assertCommandLine(['a/?.c'], 'a\\?.c')
        self.assertCommandLine(['a\\foo.c'], 'a\\foo.c')

    def test_single_quote_support(self):
        self.assertCommandLine(['add', "let's-do-it.txt"], "add let's-do-it.txt", ['add', "let's-do-it.txt"])
        self.expectFailure('Using single quotes breaks trimming from argv', self.assertCommandLine, ['add', 'lets do it.txt'], "add 'lets do it.txt'", ['add', "'lets", 'do', "it.txt'"], single_quotes_allowed=True)

    def test_case_insensitive_globs(self):
        if os.path.normcase('AbC') == 'AbC':
            self.skipTest('Test requires case insensitive globbing function')
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine(['A/b.c'], 'A/B*')

    def test_backslashes(self):
        self.requireFeature(backslashdir_feature)
        self.build_tree(['a/', 'a/b.c', 'a/c.c', 'a/c.h'])
        self.assertCommandLine(['a/b.c'], 'a\\b*')

    def test_with_pdb(self):
        """Check stripping Python arguments before bzr script per lp:587868"""
        self.assertCommandLine(['rocks'], '-m pdb rocks', ['rocks'])
        self.build_tree(['d/', 'd/f1', 'd/f2'])
        self.assertCommandLine(['rm', 'x*'], '-m pdb rm x*', ['rm', 'x*'])
        self.assertCommandLine(['add', 'd/f1', 'd/f2'], '-m pdb add d/*', ['add', 'd/*'])