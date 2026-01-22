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
class _CCExtension(Extension):
    """
    A Numba-specific Extension subclass to LLVM-compile pure Python code
    to an extension module.
    """
    _cc = None
    _distutils_monkey_patched = False

    def _prepare_object_files(self, build_ext):
        cc = self._cc
        dir_util.mkpath(os.path.join(build_ext.build_temp, *self.name.split('.')[:-1]))
        objects, _ = cc._compile_object_files(build_ext.build_temp)
        self.extra_objects = objects

    @classmethod
    def monkey_patch_distutils(cls):
        """
        Monkey-patch distutils with our own build_ext class knowing
        about pycc-compiled extensions modules.
        """
        if cls._distutils_monkey_patched:
            return
        _orig_build_ext = build_ext.build_ext

        class _CC_build_ext(_orig_build_ext):

            def build_extension(self, ext):
                if isinstance(ext, _CCExtension):
                    ext._prepare_object_files(self)
                _orig_build_ext.build_extension(self, ext)
        build_ext.build_ext = _CC_build_ext
        cls._distutils_monkey_patched = True