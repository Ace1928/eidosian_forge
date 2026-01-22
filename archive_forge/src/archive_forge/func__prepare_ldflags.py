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
def _prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
    if IS_WINDOWS:
        python_path = os.path.dirname(sys.executable)
        python_lib_path = os.path.join(python_path, 'libs')
        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_cuda.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_cuda.lib')
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
        extra_ldflags.append('torch.lib')
        extra_ldflags.append(f'/LIBPATH:{TORCH_LIB_PATH}')
        if not is_standalone:
            extra_ldflags.append('torch_python.lib')
            extra_ldflags.append(f'/LIBPATH:{python_lib_path}')
    else:
        extra_ldflags.append(f'-L{TORCH_LIB_PATH}')
        extra_ldflags.append('-lc10')
        if with_cuda:
            extra_ldflags.append('-lc10_hip' if IS_HIP_EXTENSION else '-lc10_cuda')
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')
        if is_standalone and 'TBB' in torch.__config__.parallel_info():
            extra_ldflags.append('-ltbb')
        if is_standalone:
            extra_ldflags.append(f'-Wl,-rpath,{TORCH_LIB_PATH}')
    if with_cuda:
        if verbose:
            print('Detected CUDA files, patching ldflags', file=sys.stderr)
        if IS_WINDOWS:
            extra_ldflags.append(f'/LIBPATH:{_join_cuda_home('lib', 'x64')}')
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'/LIBPATH:{os.path.join(CUDNN_HOME, 'lib', 'x64')}')
        elif not IS_HIP_EXTENSION:
            extra_lib_dir = 'lib64'
            if not os.path.exists(_join_cuda_home(extra_lib_dir)) and os.path.exists(_join_cuda_home('lib')):
                extra_lib_dir = 'lib'
            extra_ldflags.append(f'-L{_join_cuda_home(extra_lib_dir)}')
            extra_ldflags.append('-lcudart')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f'-L{os.path.join(CUDNN_HOME, 'lib64')}')
        elif IS_HIP_EXTENSION:
            assert ROCM_VERSION is not None
            extra_ldflags.append(f'-L{_join_rocm_home('lib')}')
            extra_ldflags.append('-lamdhip64' if ROCM_VERSION >= (3, 5) else '-lhip_hcc')
    return extra_ldflags