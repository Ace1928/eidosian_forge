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
def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
    cflags = copy.deepcopy(extra_postargs)
    try:
        original_compiler = self.compiler.compiler_so
        if _is_cuda_file(src):
            nvcc = [_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc')]
            self.compiler.set_executable('compiler_so', nvcc)
            if isinstance(cflags, dict):
                cflags = cflags['nvcc']
            if IS_HIP_EXTENSION:
                cflags = COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
            else:
                cflags = unix_cuda_flags(cflags)
        elif isinstance(cflags, dict):
            cflags = cflags['cxx']
        if IS_HIP_EXTENSION:
            cflags = COMMON_HIP_FLAGS + cflags
        append_std17_if_no_std_present(cflags)
        original_compile(obj, src, ext, cc_args, cflags, pp_opts)
    finally:
        self.compiler.set_executable('compiler_so', original_compiler)