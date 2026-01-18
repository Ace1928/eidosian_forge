import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections
from pathlib import Path
import errno
import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
from typing import Dict, List, Optional, Union, Tuple
from torch.torch_version import TorchVersion, Version
from setuptools.command.build_ext import build_ext
def win_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    if not self.compiler.initialized:
        self.compiler.initialize()
    output_dir = os.path.abspath(output_dir)
    convert_to_absolute_paths_inplace(self.compiler.include_dirs)
    _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    common_cflags = extra_preargs or []
    cflags = []
    if debug:
        cflags.extend(self.compiler.compile_options_debug)
    else:
        cflags.extend(self.compiler.compile_options)
    common_cflags.extend(COMMON_MSVC_FLAGS)
    cflags = cflags + common_cflags + pp_opts
    with_cuda = any(map(_is_cuda_file, sources))
    if isinstance(extra_postargs, dict):
        post_cflags = extra_postargs['cxx']
    else:
        post_cflags = list(extra_postargs)
    append_std17_if_no_std_present(post_cflags)
    cuda_post_cflags = None
    cuda_cflags = None
    if with_cuda:
        cuda_cflags = ['-std=c++17', '--use-local-env']
        for common_cflag in common_cflags:
            cuda_cflags.append('-Xcompiler')
            cuda_cflags.append(common_cflag)
        for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
            cuda_cflags.append('-Xcudafe')
            cuda_cflags.append('--diag_suppress=' + ignore_warning)
        cuda_cflags.extend(pp_opts)
        if isinstance(extra_postargs, dict):
            cuda_post_cflags = extra_postargs['nvcc']
        else:
            cuda_post_cflags = list(extra_postargs)
        cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
    cflags = _nt_quote_args(cflags)
    post_cflags = _nt_quote_args(post_cflags)
    if with_cuda:
        cuda_cflags = _nt_quote_args(cuda_cflags)
        cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
    if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
        cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
    else:
        cuda_dlink_post_cflags = None
    _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, cuda_dlink_post_cflags=cuda_dlink_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
    return objects