from distutils import errors
import imp
import os
import re
import shlex
import sys
import traceback
from setuptools.command import test
class GoogleTest(test.test):
    """Command to run Google-style tests after in-place build."""
    description = 'run Google-style tests after in-place build'
    _DEFAULT_PATTERN = '_(?:unit|reg)?test\\.py$'
    user_options = [('test-dir=', 'd', 'Look for test modules in specified directory.'), ('test-module-pattern=', 'p', 'Pattern for matching test modules. Defaults to %r. Only source files (*.py) will be considered, even if more files match this pattern.' % _DEFAULT_PATTERN), ('test-args=', 'a', 'Arguments to pass to basetest.main(). May only make sense if test_module_pattern matches exactly one test.')]

    def initialize_options(self):
        self.test_dir = None
        self.test_module_pattern = self._DEFAULT_PATTERN
        self.test_args = ''
        self.test_suite = True

    def finalize_options(self):
        if self.test_dir is None:
            if self.distribution.google_test_dir:
                self.test_dir = self.distribution.google_test_dir
            else:
                raise errors.DistutilsOptionError('No test directory specified')
        self.test_module_pattern = re.compile(self.test_module_pattern)
        self.test_args = shlex.split(self.test_args)

    def _RunTestModule(self, module_path):
        """Run a module as a test module given its path.

    Args:
      module_path: The path to the module to test; must end in '.py'.

    Returns:
      True if the tests in this module pass, False if not or if an error occurs.
    """
        path, filename = os.path.split(module_path)
        old_argv = sys.argv[:]
        old_path = sys.path[:]
        old_modules = sys.modules.copy()
        sys.path.insert(0, path)
        module_name = filename.replace('.py', '')
        import_tuple = imp.find_module(module_name, [path])
        module = imp.load_module(module_name, *import_tuple)
        sys.modules['__main__'] = module
        sys.argv = [module.__file__] + self.test_args
        import basetest
        try:
            try:
                sys.stderr.write('Testing %s\n' % module_name)
                basetest.main()
                return False
            except SystemExit as e:
                returncode, = e.args
                return not returncode
            except:
                traceback.print_exc()
                return False
        finally:
            sys.argv[:] = old_argv
            sys.path[:] = old_path
            sys.modules.clear()
            sys.modules.update(old_modules)

    def run_tests(self):
        ok = True
        for path, _, filenames in os.walk(self.test_dir):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                file_path = os.path.join(path, filename)
                if self.test_module_pattern.search(file_path):
                    ok &= self._RunTestModule(file_path)
        sys.exit(int(not ok))