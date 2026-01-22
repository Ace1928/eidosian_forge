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
def _get_extra_cflags(self):
    extra_cflags = self._extra_cflags.get(sys.platform, [])
    if not extra_cflags:
        extra_cflags = self._extra_cflags.get(os.name, [])
    return extra_cflags