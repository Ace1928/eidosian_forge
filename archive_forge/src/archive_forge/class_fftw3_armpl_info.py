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
class fftw3_armpl_info(fftw_info):
    section = 'fftw3'
    dir_env_var = 'ARMPL_DIR'
    notfounderror = FFTWNotFoundError
    ver_info = [{'name': 'fftw3', 'libs': ['armpl_lp64_mp'], 'includes': ['fftw3.h'], 'macros': [('SCIPY_FFTW3_H', None)]}]