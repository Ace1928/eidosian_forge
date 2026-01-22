import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def _calc_info_blas(self):
    warnings.warn(BlasOptNotFoundError.__doc__ or '', stacklevel=3)
    info = {}
    dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])
    blas = get_info('blas')
    if blas:
        dict_append(info, **blas)
    else:
        warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=3)
        blas_src = get_info('blas_src')
        if not blas_src:
            warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=3)
            return False
        dict_append(info, libraries=[('fblas_src', blas_src)])
    self.set_info(**info)
    return True