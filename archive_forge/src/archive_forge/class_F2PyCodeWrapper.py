import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup
from numpy import get_include
class F2PyCodeWrapper(CodeWrapper):
    """Wrapper that uses f2py"""

    def __init__(self, *args, **kwargs):
        ext_keys = ['include_dirs', 'library_dirs', 'libraries', 'extra_compile_args', 'extra_link_args']
        msg = 'The compilation option kwarg {} is not supported with the f2py backend.'
        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)
        super().__init__(*args, **kwargs)

    @property
    def command(self):
        filename = self.filename + '.' + self.generator.code_extension
        args = ['-c', '-m', self.module_name, filename]
        command = [sys.executable, '-c', 'import numpy.f2py as f2py2e;f2py2e.main()'] + args
        return command

    def _prepare_files(self, routine):
        pass

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)