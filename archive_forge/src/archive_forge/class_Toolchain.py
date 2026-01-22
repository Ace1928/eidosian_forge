import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
class Toolchain(object):

    def __init__(self):
        if not external_compiler_works():
            self._raise_external_compiler_error()
        self._verbose = False
        self._compiler = new_compiler()
        customize_compiler(self._compiler)
        self._build_ext = build_ext(Distribution())
        self._build_ext.finalize_options()
        self._py_lib_dirs = self._build_ext.library_dirs
        self._py_include_dirs = self._build_ext.include_dirs
        np_compile_args = {'include_dirs': [np.get_include()]}
        if sys.platform == 'win32':
            np_compile_args['libraries'] = []
        else:
            np_compile_args['libraries'] = ['m']
        self._math_info = np_compile_args

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        log.set_threshold(log.INFO if value else log.WARN)

    def _raise_external_compiler_error(self):
        basemsg = 'Attempted to compile AOT function without the compiler used by `numpy.distutils` present.'
        conda_msg = 'If using conda try:\n\n#> conda install %s'
        plt = sys.platform
        if plt.startswith('linux'):
            if sys.maxsize <= 2 ** 32:
                compilers = ['gcc_linux-32', 'gxx_linux-32']
            else:
                compilers = ['gcc_linux-64', 'gxx_linux-64']
            msg = '%s %s' % (basemsg, conda_msg % ' '.join(compilers))
        elif plt.startswith('darwin'):
            compilers = ['clang_osx-64', 'clangxx_osx-64']
            msg = '%s %s' % (basemsg, conda_msg % ' '.join(compilers))
        elif plt.startswith('win32'):
            winmsg = 'Cannot find suitable msvc.'
            msg = '%s %s' % (basemsg, winmsg)
        else:
            msg = 'Unknown platform %s' % plt
        raise RuntimeError(msg)

    def compile_objects(self, sources, output_dir, include_dirs=(), depends=(), macros=(), extra_cflags=None):
        """
        Compile the given source files into a separate object file each,
        all beneath the *output_dir*.  A list of paths to object files
        is returned.

        *macros* has the same format as in distutils: a list of 1- or 2-tuples.
        If a 1-tuple (name,), the given name is considered undefined by
        the C preprocessor.
        If a 2-tuple (name, value), the given name is expanded into the
        given value by the C preprocessor.
        """
        objects = self._compiler.compile(sources, output_dir=output_dir, include_dirs=include_dirs, depends=depends, macros=macros or [], extra_preargs=extra_cflags)
        return objects

    def link_shared(self, output, objects, libraries=(), library_dirs=(), export_symbols=(), extra_ldflags=None):
        """
        Create a shared library *output* linking the given *objects*
        and *libraries* (all strings).
        """
        output_dir, output_filename = os.path.split(output)
        self._compiler.link(CCompiler.SHARED_OBJECT, objects, output_filename, output_dir, libraries, library_dirs, export_symbols=export_symbols, extra_preargs=extra_ldflags)

    def get_python_libraries(self):
        """
        Get the library arguments necessary to link with Python.
        """
        libs = self._build_ext.get_libraries(_DummyExtension())
        if sys.platform == 'win32':
            libs = libs + ['msvcrt']
        return libs + self._math_info['libraries']

    def get_python_library_dirs(self):
        """
        Get the library directories necessary to link with Python.
        """
        return list(self._py_lib_dirs)

    def get_python_include_dirs(self):
        """
        Get the include directories necessary to compile against the Python
        and Numpy C APIs.
        """
        return list(self._py_include_dirs) + self._math_info['include_dirs']

    def get_ext_filename(self, ext_name):
        """
        Given a C extension's module name, return its intended filename.
        """
        return self._build_ext.get_ext_filename(ext_name)