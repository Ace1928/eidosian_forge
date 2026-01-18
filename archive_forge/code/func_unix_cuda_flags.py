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
def unix_cuda_flags(cflags):
    cflags = COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] + cflags + _get_cuda_arch_flags(cflags)
    _ccbin = os.getenv('CC')
    if _ccbin is not None and (not any((flag.startswith(('-ccbin', '--compiler-bindir')) for flag in cflags))):
        cflags.extend(['-ccbin', _ccbin])
    return cflags