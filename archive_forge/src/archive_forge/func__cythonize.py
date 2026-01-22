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
def _cythonize(self, module_name, code, lib_dir, args, quiet=True):
    pyx_file = os.path.join(lib_dir, module_name + '.pyx')
    pyx_file = encode_fs(pyx_file)
    c_include_dirs = args.include
    c_src_files = list(map(str, args.src))
    if 'numpy' in code:
        import numpy
        c_include_dirs.append(numpy.get_include())
    with io.open(pyx_file, 'w', encoding='utf-8') as f:
        f.write(code)
    extension = Extension(name=module_name, sources=[pyx_file] + c_src_files, include_dirs=c_include_dirs, library_dirs=args.library_dirs, extra_compile_args=args.compile_args, extra_link_args=args.link_args, libraries=args.lib, language='c++' if args.cplus else 'c')
    try:
        opts = dict(quiet=quiet, annotate=args.annotate, force=True, language_level=min(3, sys.version_info[0]))
        if args.language_level is not None:
            assert args.language_level in (2, 3)
            opts['language_level'] = args.language_level
        return cythonize([extension], **opts)
    except CompileError:
        return None