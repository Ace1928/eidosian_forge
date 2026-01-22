from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
class TestParseArgs(TestCase):

    def setUp(self):
        self._options_backup = backup_Options()

    def tearDown(self):
        restore_Options(self._options_backup)

    def check_default_global_options(self, white_list=[]):
        self.assertEqual(check_global_options(self._options_backup, white_list), '')

    def test_build_set_for_inplace(self):
        options, args = parse_args(['foo.pyx', '-i'])
        self.assertEqual(options.build, True)
        self.check_default_global_options()

    def test_lenient(self):
        options, sources = parse_args(['foo.pyx', '--lenient'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.error_on_unknown_names, False)
        self.assertEqual(Options.error_on_uninitialized, False)
        self.check_default_global_options(['error_on_unknown_names', 'error_on_uninitialized'])

    def test_annotate(self):
        options, sources = parse_args(['foo.pyx', '--annotate'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.annotate, 'default')
        self.check_default_global_options(['annotate'])

    def test_annotate_fullc(self):
        options, sources = parse_args(['foo.pyx', '--annotate-fullc'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.annotate, 'fullc')
        self.check_default_global_options(['annotate'])

    def test_no_docstrings(self):
        options, sources = parse_args(['foo.pyx', '--no-docstrings'])
        self.assertEqual(sources, ['foo.pyx'])
        self.assertEqual(Options.docstrings, False)
        self.check_default_global_options(['docstrings'])