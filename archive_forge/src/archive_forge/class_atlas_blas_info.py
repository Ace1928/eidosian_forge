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
class atlas_blas_info(atlas_info):
    _lib_names = ['f77blas', 'cblas']

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        info = {}
        opt = self.get_option_single('atlas_libs', 'libraries')
        atlas_libs = self.get_libs(opt, self._lib_names + self._lib_atlas)
        atlas = self.check_libs2(lib_dirs, atlas_libs, [])
        if atlas is None:
            return
        include_dirs = self.get_include_dirs()
        h = self.combine_paths(lib_dirs + include_dirs, 'cblas.h') or [None]
        h = h[0]
        if h:
            h = os.path.dirname(h)
            dict_append(info, include_dirs=[h])
        info['language'] = 'c'
        info['define_macros'] = [('HAVE_CBLAS', None)]
        atlas_version, atlas_extra_info = get_atlas_version(**atlas)
        dict_append(atlas, **atlas_extra_info)
        dict_append(info, **atlas)
        self.set_info(**info)
        return