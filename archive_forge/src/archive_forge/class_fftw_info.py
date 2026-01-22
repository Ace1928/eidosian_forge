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
class fftw_info(system_info):
    section = 'fftw'
    dir_env_var = 'FFTW'
    notfounderror = FFTWNotFoundError
    ver_info = [{'name': 'fftw3', 'libs': ['fftw3'], 'includes': ['fftw3.h'], 'macros': [('SCIPY_FFTW3_H', None)]}, {'name': 'fftw2', 'libs': ['rfftw', 'fftw'], 'includes': ['fftw.h', 'rfftw.h'], 'macros': [('SCIPY_FFTW_H', None)]}]

    def calc_ver_info(self, ver_param):
        """Returns True on successful version detection, else False"""
        lib_dirs = self.get_lib_dirs()
        incl_dirs = self.get_include_dirs()
        opt = self.get_option_single(self.section + '_libs', 'libraries')
        libs = self.get_libs(opt, ver_param['libs'])
        info = self.check_libs(lib_dirs, libs)
        if info is not None:
            flag = 0
            for d in incl_dirs:
                if len(self.combine_paths(d, ver_param['includes'])) == len(ver_param['includes']):
                    dict_append(info, include_dirs=[d])
                    flag = 1
                    break
            if flag:
                dict_append(info, define_macros=ver_param['macros'])
            else:
                info = None
        if info is not None:
            self.set_info(**info)
            return True
        else:
            log.info('  %s not found' % ver_param['name'])
            return False

    def calc_info(self):
        for i in self.ver_info:
            if self.calc_ver_info(i):
                break