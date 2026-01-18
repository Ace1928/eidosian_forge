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
def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    """Compiles sources by outputting a ninja file and running it."""
    output_dir = os.path.abspath(output_dir)
    convert_to_absolute_paths_inplace(self.compiler.include_dirs)
    _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
    extra_cc_cflags = self.compiler.compiler_so[1:]
    with_cuda = any(map(_is_cuda_file, sources))
    if isinstance(extra_postargs, dict):
        post_cflags = extra_postargs['cxx']
    else:
        post_cflags = list(extra_postargs)
    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    append_std17_if_no_std_present(post_cflags)
    cuda_post_cflags = None
    cuda_cflags = None
    if with_cuda:
        cuda_cflags = common_cflags
        if isinstance(extra_postargs, dict):
            cuda_post_cflags = extra_postargs['nvcc']
        else:
            cuda_post_cflags = list(extra_postargs)
        if IS_HIP_EXTENSION:
            cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
            cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
        else:
            cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
        append_std17_if_no_std_present(cuda_post_cflags)
        cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
        cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]
    if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
        cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
    else:
        cuda_dlink_post_cflags = None
    _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags], post_cflags=[shlex.quote(f) for f in post_cflags], cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, cuda_dlink_post_cflags=cuda_dlink_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
    return objects