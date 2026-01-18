from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_parallel_long(self):
    options, args = self.parse_args(['--parallel', '42'])
    self.assertFalse(args)
    self.assertTrue(self.are_default(options, ['parallel']))
    self.assertEqual(options.parallel, 42)