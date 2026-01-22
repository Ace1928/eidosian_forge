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
class blas_opt_info(system_info):
    notfounderror = BlasNotFoundError
    blas_order = ['armpl', 'mkl', 'ssl2', 'blis', 'openblas', 'accelerate', 'atlas', 'blas']
    order_env_var_name = 'NPY_BLAS_ORDER'

    def _calc_info_armpl(self):
        info = get_info('blas_armpl')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_mkl(self):
        info = get_info('blas_mkl')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_ssl2(self):
        info = get_info('blas_ssl2')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_blis(self):
        info = get_info('blis')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_openblas(self):
        info = get_info('openblas')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_atlas(self):
        info = get_info('atlas_3_10_blas_threads')
        if not info:
            info = get_info('atlas_3_10_blas')
        if not info:
            info = get_info('atlas_blas_threads')
        if not info:
            info = get_info('atlas_blas')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_accelerate(self):
        info = get_info('accelerate')
        if info:
            self.set_info(**info)
            return True
        return False

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

    def _calc_info_from_envvar(self):
        info = {}
        info['language'] = 'f77'
        info['libraries'] = []
        info['include_dirs'] = []
        info['define_macros'] = []
        info['extra_link_args'] = os.environ['NPY_BLAS_LIBS'].split()
        if 'NPY_CBLAS_LIBS' in os.environ:
            info['define_macros'].append(('HAVE_CBLAS', None))
            info['extra_link_args'].extend(os.environ['NPY_CBLAS_LIBS'].split())
        self.set_info(**info)
        return True

    def _calc_info(self, name):
        return getattr(self, '_calc_info_{}'.format(name))()

    def calc_info(self):
        blas_order, unknown_order = _parse_env_order(self.blas_order, self.order_env_var_name)
        if len(unknown_order) > 0:
            raise ValueError('blas_opt_info user defined BLAS order has unacceptable values: {}'.format(unknown_order))
        if 'NPY_BLAS_LIBS' in os.environ:
            self._calc_info_from_envvar()
            return
        for blas in blas_order:
            if self._calc_info(blas):
                return
        if 'blas' not in blas_order:
            warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=2)
            warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=2)