from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def update_pythran_extension(ext):
    if pythran is None:
        raise RuntimeError('You first need to install Pythran to use the np_pythran directive.')
    try:
        pythran_ext = pythran.config.make_extension(python=True)
    except TypeError:
        pythran_ext = pythran.config.make_extension()
    ext.include_dirs.extend(pythran_ext['include_dirs'])
    ext.extra_compile_args.extend(pythran_ext['extra_compile_args'])
    ext.extra_link_args.extend(pythran_ext['extra_link_args'])
    ext.define_macros.extend(pythran_ext['define_macros'])
    ext.undef_macros.extend(pythran_ext['undef_macros'])
    ext.library_dirs.extend(pythran_ext['library_dirs'])
    ext.libraries.extend(pythran_ext['libraries'])
    ext.language = 'c++'
    for bad_option in ['-fwhole-program', '-fvisibility=hidden']:
        try:
            ext.extra_compile_args.remove(bad_option)
        except ValueError:
            pass