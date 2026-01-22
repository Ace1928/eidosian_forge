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
class ssl2_info(system_info):
    section = 'ssl2'
    dir_env_var = 'SSL2_DIR'
    _lib_ssl2 = ['fjlapackexsve']

    def get_tcsds_rootdir(self):
        tcsdsroot = os.environ.get('TCSDS_PATH', None)
        if tcsdsroot is not None:
            return tcsdsroot
        return None

    def __init__(self):
        tcsdsroot = self.get_tcsds_rootdir()
        if tcsdsroot is None:
            system_info.__init__(self)
        else:
            system_info.__init__(self, default_lib_dirs=[os.path.join(tcsdsroot, 'lib64')], default_include_dirs=[os.path.join(tcsdsroot, 'clang-comp/include')])

    def calc_info(self):
        tcsdsroot = self.get_tcsds_rootdir()
        lib_dirs = self.get_lib_dirs()
        if lib_dirs is None:
            lib_dirs = os.path.join(tcsdsroot, 'lib64')
        incl_dirs = self.get_include_dirs()
        if incl_dirs is None:
            incl_dirs = os.path.join(tcsdsroot, 'clang-comp/include')
        ssl2_libs = self.get_libs('ssl2_libs', self._lib_ssl2)
        info = self.check_libs2(lib_dirs, ssl2_libs)
        if info is None:
            return
        dict_append(info, define_macros=[('HAVE_CBLAS', None), ('HAVE_SSL2', 1)], include_dirs=incl_dirs)
        self.set_info(**info)