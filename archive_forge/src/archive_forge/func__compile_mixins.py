from setuptools import distutils as dutils
from setuptools.command import build_ext
from setuptools.extension import Extension
import os
import shutil
import sys
import tempfile
from numba.core import typing, sigutils
from numba.core.compiler_lock import global_compiler_lock
from numba.pycc.compiler import ModuleCompiler, ExportEntry
from numba.pycc.platform import Toolchain
from numba import cext
def _compile_mixins(self, build_dir):
    sources = self._get_mixin_sources()
    macros = self._get_mixin_defines()
    include_dirs = self._toolchain.get_python_include_dirs()
    extra_cflags = self._get_extra_cflags()
    objects = self._toolchain.compile_objects(sources, build_dir, include_dirs=include_dirs, macros=macros, extra_cflags=extra_cflags)
    return objects