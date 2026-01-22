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
def _add_pgo_flags(self, build_extension, step_name, temp_dir):
    compiler_type = build_extension.compiler.compiler_type
    if compiler_type == 'unix':
        compiler_cmd = build_extension.compiler.compiler_so
        if not compiler_cmd:
            pass
        elif 'clang' in compiler_cmd or 'clang' in compiler_cmd[0]:
            compiler_type = 'clang'
        elif 'icc' in compiler_cmd or 'icc' in compiler_cmd[0]:
            compiler_type = 'icc'
        elif 'gcc' in compiler_cmd or 'gcc' in compiler_cmd[0]:
            compiler_type = 'gcc'
        elif 'g++' in compiler_cmd or 'g++' in compiler_cmd[0]:
            compiler_type = 'gcc'
    config = PGO_CONFIG.get(compiler_type)
    orig_flags = []
    if config and step_name in config:
        flags = [f.format(TEMPDIR=temp_dir) for f in config[step_name]]
        for extension in build_extension.extensions:
            orig_flags.append((extension.extra_compile_args, extension.extra_link_args))
            extension.extra_compile_args = extension.extra_compile_args + flags
            extension.extra_link_args = extension.extra_link_args + flags
    else:
        print("No PGO %s configuration known for C compiler type '%s'" % (step_name, compiler_type), file=sys.stderr)
    return orig_flags