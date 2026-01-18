from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_option_multiple(self):
    options, args = self.parse_args(['-s', 'docstrings=True', '-s', 'buffer_max_dims=8'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['options']))
    self.assertEqual(options.options['docstrings'], True)
    self.assertEqual(options.options['buffer_max_dims'], True)