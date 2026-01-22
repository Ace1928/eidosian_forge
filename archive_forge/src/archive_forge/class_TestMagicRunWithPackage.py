import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
class TestMagicRunWithPackage(unittest.TestCase):

    def writefile(self, name, content):
        path = os.path.join(self.tempdir.name, name)
        d = os.path.dirname(path)
        if not os.path.isdir(d):
            os.makedirs(d)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(textwrap.dedent(content))

    def setUp(self):
        self.package = package = 'tmp{0}'.format(''.join([random.choice(string.ascii_letters) for i in range(10)]))
        'Temporary  (probably) valid python package name.'
        self.value = int(random.random() * 10000)
        self.tempdir = TemporaryDirectory()
        self.__orig_cwd = os.getcwd()
        sys.path.insert(0, self.tempdir.name)
        self.writefile(os.path.join(package, '__init__.py'), '')
        self.writefile(os.path.join(package, 'sub.py'), '\n        x = {0!r}\n        '.format(self.value))
        self.writefile(os.path.join(package, 'relative.py'), '\n        from .sub import x\n        ')
        self.writefile(os.path.join(package, 'absolute.py'), '\n        from {0}.sub import x\n        '.format(package))
        self.writefile(os.path.join(package, 'args.py'), '\n        import sys\n        a = " ".join(sys.argv[1:])\n        '.format(package))

    def tearDown(self):
        os.chdir(self.__orig_cwd)
        sys.path[:] = [p for p in sys.path if p != self.tempdir.name]
        self.tempdir.cleanup()

    def check_run_submodule(self, submodule, opts=''):
        _ip.user_ns.pop('x', None)
        _ip.run_line_magic('run', '{2} -m {0}.{1}'.format(self.package, submodule, opts))
        self.assertEqual(_ip.user_ns['x'], self.value, 'Variable `x` is not loaded from module `{0}`.'.format(submodule))

    def test_run_submodule_with_absolute_import(self):
        self.check_run_submodule('absolute')

    def test_run_submodule_with_relative_import(self):
        """Run submodule that has a relative import statement (#2727)."""
        self.check_run_submodule('relative')

    def test_prun_submodule_with_absolute_import(self):
        self.check_run_submodule('absolute', '-p')

    def test_prun_submodule_with_relative_import(self):
        self.check_run_submodule('relative', '-p')

    def with_fake_debugger(func):

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            with patch.object(debugger.Pdb, 'run', staticmethod(eval)):
                return func(*args, **kwds)
        return wrapper

    @with_fake_debugger
    def test_debug_run_submodule_with_absolute_import(self):
        self.check_run_submodule('absolute', '-d')

    @with_fake_debugger
    def test_debug_run_submodule_with_relative_import(self):
        self.check_run_submodule('relative', '-d')

    def test_module_options(self):
        _ip.user_ns.pop('a', None)
        test_opts = '-x abc -m test'
        _ip.run_line_magic('run', '-m {0}.args {1}'.format(self.package, test_opts))
        assert _ip.user_ns['a'] == test_opts

    def test_module_options_with_separator(self):
        _ip.user_ns.pop('a', None)
        test_opts = '-x abc -m test'
        _ip.run_line_magic('run', '-m {0}.args -- {1}'.format(self.package, test_opts))
        assert _ip.user_ns['a'] == test_opts