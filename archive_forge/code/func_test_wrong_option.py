from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def test_wrong_option(self):
    old_stderr = sys.stderr
    stderr = sys.stderr = StringIO()
    try:
        self.assertRaises(SystemExit, self.parse_args, ['--unknown-option'])
    finally:
        sys.stderr = old_stderr
    self.assertTrue(stderr.getvalue())