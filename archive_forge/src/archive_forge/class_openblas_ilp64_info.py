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
class openblas_ilp64_info(openblas_info):
    section = 'openblas_ilp64'
    dir_env_var = 'OPENBLAS_ILP64'
    _lib_names = ['openblas64']
    _require_symbols = ['dgemm_', 'cblas_dgemm']
    notfounderror = BlasILP64NotFoundError

    def _calc_info(self):
        info = super()._calc_info()
        if info is not None:
            info['define_macros'] += [('HAVE_BLAS_ILP64', None)]
        return info