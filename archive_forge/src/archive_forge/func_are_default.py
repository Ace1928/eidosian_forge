from Cython.Build.Cythonize import (
from Cython.Compiler import Options
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from unittest import TestCase
import sys
def are_default(self, options, skip):
    empty_containers = ['directives', 'compile_time_env', 'options', 'excludes']
    are_none = ['language_level', 'annotate', 'build', 'build_inplace', 'force', 'quiet', 'lenient', 'keep_going', 'no_docstrings']
    for opt_name in empty_containers:
        if len(getattr(options, opt_name)) != 0 and opt_name not in skip:
            self.assertEqual(opt_name, '', msg='For option ' + opt_name)
            return False
    for opt_name in are_none:
        if getattr(options, opt_name) is not None and opt_name not in skip:
            self.assertEqual(opt_name, '', msg='For option ' + opt_name)
            return False
    if options.parallel != parallel_compiles and 'parallel' not in skip:
        return False
    return True