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
class lapack_opt_info(system_info):
    notfounderror = LapackNotFoundError
    lapack_order = ['armpl', 'mkl', 'ssl2', 'openblas', 'flame', 'accelerate', 'atlas', 'lapack']
    order_env_var_name = 'NPY_LAPACK_ORDER'

    def _calc_info_armpl(self):
        info = get_info('lapack_armpl')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_mkl(self):
        info = get_info('lapack_mkl')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_ssl2(self):
        info = get_info('lapack_ssl2')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_openblas(self):
        info = get_info('openblas_lapack')
        if info:
            self.set_info(**info)
            return True
        info = get_info('openblas_clapack')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_flame(self):
        info = get_info('flame')
        if info:
            self.set_info(**info)
            return True
        return False

    def _calc_info_atlas(self):
        info = get_info('atlas_3_10_threads')
        if not info:
            info = get_info('atlas_3_10')
        if not info:
            info = get_info('atlas_threads')
        if not info:
            info = get_info('atlas')
        if info:
            l = info.get('define_macros', [])
            if ('ATLAS_WITH_LAPACK_ATLAS', None) in l or ('ATLAS_WITHOUT_LAPACK', None) in l:
                lapack_info = self._get_info_lapack()
                if not lapack_info:
                    return False
                dict_append(info, **lapack_info)
            self.set_info(**info)
            return True
        return False

    def _calc_info_accelerate(self):
        info = get_info('accelerate')
        if info:
            self.set_info(**info)
            return True
        return False

    def _get_info_blas(self):
        info = get_info('blas_opt')
        if not info:
            warnings.warn(BlasNotFoundError.__doc__ or '', stacklevel=3)
            info_src = get_info('blas_src')
            if not info_src:
                warnings.warn(BlasSrcNotFoundError.__doc__ or '', stacklevel=3)
                return {}
            dict_append(info, libraries=[('fblas_src', info_src)])
        return info

    def _get_info_lapack(self):
        info = get_info('lapack')
        if not info:
            warnings.warn(LapackNotFoundError.__doc__ or '', stacklevel=3)
            info_src = get_info('lapack_src')
            if not info_src:
                warnings.warn(LapackSrcNotFoundError.__doc__ or '', stacklevel=3)
                return {}
            dict_append(info, libraries=[('flapack_src', info_src)])
        return info

    def _calc_info_lapack(self):
        info = self._get_info_lapack()
        if info:
            info_blas = self._get_info_blas()
            dict_append(info, **info_blas)
            dict_append(info, define_macros=[('NO_ATLAS_INFO', 1)])
            self.set_info(**info)
            return True
        return False

    def _calc_info_from_envvar(self):
        info = {}
        info['language'] = 'f77'
        info['libraries'] = []
        info['include_dirs'] = []
        info['define_macros'] = []
        info['extra_link_args'] = os.environ['NPY_LAPACK_LIBS'].split()
        self.set_info(**info)
        return True

    def _calc_info(self, name):
        return getattr(self, '_calc_info_{}'.format(name))()

    def calc_info(self):
        lapack_order, unknown_order = _parse_env_order(self.lapack_order, self.order_env_var_name)
        if len(unknown_order) > 0:
            raise ValueError('lapack_opt_info user defined LAPACK order has unacceptable values: {}'.format(unknown_order))
        if 'NPY_LAPACK_LIBS' in os.environ:
            self._calc_info_from_envvar()
            return
        for lapack in lapack_order:
            if self._calc_info(lapack):
                return
        if 'lapack' not in lapack_order:
            warnings.warn(LapackNotFoundError.__doc__ or '', stacklevel=2)
            warnings.warn(LapackSrcNotFoundError.__doc__ or '', stacklevel=2)