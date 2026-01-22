from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
def _build_extension(self, extension, lib_dir, temp_dir=None, pgo_step_name=None, quiet=True):
    build_extension = self._get_build_extension(extension, lib_dir=lib_dir, temp_dir=temp_dir, pgo_step_name=pgo_step_name)
    old_threshold = None
    try:
        if not quiet:
            old_threshold = distutils.log.set_threshold(distutils.log.DEBUG)
        build_extension.run()
    finally:
        if not quiet and old_threshold is not None:
            distutils.log.set_threshold(old_threshold)