import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
def _raise_external_compiler_error(self):
    basemsg = 'Attempted to compile AOT function without the compiler used by `numpy.distutils` present.'
    conda_msg = 'If using conda try:\n\n#> conda install %s'
    plt = sys.platform
    if plt.startswith('linux'):
        if sys.maxsize <= 2 ** 32:
            compilers = ['gcc_linux-32', 'gxx_linux-32']
        else:
            compilers = ['gcc_linux-64', 'gxx_linux-64']
        msg = '%s %s' % (basemsg, conda_msg % ' '.join(compilers))
    elif plt.startswith('darwin'):
        compilers = ['clang_osx-64', 'clangxx_osx-64']
        msg = '%s %s' % (basemsg, conda_msg % ' '.join(compilers))
    elif plt.startswith('win32'):
        winmsg = 'Cannot find suitable msvc.'
        msg = '%s %s' % (basemsg, winmsg)
    else:
        msg = 'Unknown platform %s' % plt
    raise RuntimeError(msg)