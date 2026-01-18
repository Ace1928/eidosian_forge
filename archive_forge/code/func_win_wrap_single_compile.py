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
def win_wrap_single_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    self.cflags = copy.deepcopy(extra_postargs)
    extra_postargs = None

    def spawn(cmd):
        src_regex = re.compile('/T(p|c)(.*)')
        src_list = [m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m]
        obj_regex = re.compile('/Fo(.*)')
        obj_list = [m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m]
        include_regex = re.compile('((\\-|\\/)I.*)')
        include_list = [m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m]
        if len(src_list) >= 1 and len(obj_list) >= 1:
            src = src_list[0]
            obj = obj_list[0]
            if _is_cuda_file(src):
                nvcc = _join_cuda_home('bin', 'nvcc')
                if isinstance(self.cflags, dict):
                    cflags = self.cflags['nvcc']
                elif isinstance(self.cflags, list):
                    cflags = self.cflags
                else:
                    cflags = []
                cflags = win_cuda_flags(cflags) + ['-std=c++17', '--use-local-env']
                for flag in COMMON_MSVC_FLAGS:
                    cflags = ['-Xcompiler', flag] + cflags
                for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                    cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
            elif isinstance(self.cflags, dict):
                cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                append_std17_if_no_std_present(cflags)
                cmd += cflags
            elif isinstance(self.cflags, list):
                cflags = COMMON_MSVC_FLAGS + self.cflags
                append_std17_if_no_std_present(cflags)
                cmd += cflags
        return original_spawn(cmd)
    try:
        self.compiler.spawn = spawn
        return original_compile(sources, output_dir, macros, include_dirs, debug, extra_preargs, extra_postargs, depends)
    finally:
        self.compiler.spawn = original_spawn