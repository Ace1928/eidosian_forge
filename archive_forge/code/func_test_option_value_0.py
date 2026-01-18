from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_option_value_0(self):
    options, args = self.parse_args(['-s', 'docstrings=0'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['options']))
    self.assertEqual(options.options['docstrings'], False)