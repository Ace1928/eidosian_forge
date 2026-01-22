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
class boost_python_info(system_info):
    section = 'boost_python'
    dir_env_var = 'BOOST'

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend([d] + self.combine_paths(d, ['boost*']))
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        src_dirs = self.get_src_dirs()
        src_dir = ''
        for d in src_dirs:
            if os.path.isfile(os.path.join(d, 'libs', 'python', 'src', 'module.cpp')):
                src_dir = d
                break
        if not src_dir:
            return
        py_incl_dirs = [sysconfig.get_path('include')]
        py_pincl_dir = sysconfig.get_path('platinclude')
        if py_pincl_dir not in py_incl_dirs:
            py_incl_dirs.append(py_pincl_dir)
        srcs_dir = os.path.join(src_dir, 'libs', 'python', 'src')
        bpl_srcs = glob(os.path.join(srcs_dir, '*.cpp'))
        bpl_srcs += glob(os.path.join(srcs_dir, '*', '*.cpp'))
        info = {'libraries': [('boost_python_src', {'include_dirs': [src_dir] + py_incl_dirs, 'sources': bpl_srcs})], 'include_dirs': [src_dir]}
        if info:
            self.set_info(**info)
        return