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
def _get_extra_ldflags(self):
    extra_ldflags = self._extra_ldflags.get(sys.platform, [])
    if not extra_ldflags:
        extra_ldflags = self._extra_ldflags.get(os.name, [])
    if sys.platform.startswith('linux'):
        if '-pthread' not in extra_ldflags:
            extra_ldflags.append('-pthread')
    return extra_ldflags