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
class atlas_info(system_info):
    section = 'atlas'
    dir_env_var = 'ATLAS'
    _lib_names = ['f77blas', 'cblas']
    if sys.platform[:7] == 'freebsd':
        _lib_atlas = ['atlas_r']
        _lib_lapack = ['alapack_r']
    else:
        _lib_atlas = ['atlas']
        _lib_lapack = ['lapack']
    notfounderror = AtlasNotFoundError

    def get_paths(self, section, key):
        pre_dirs = system_info.get_paths(self, section, key)
        dirs = []
        for d in pre_dirs:
            dirs.extend(self.combine_paths(d, ['atlas*', 'ATLAS*', 'sse', '3dnow', 'sse2']) + [d])
        return [d for d in dirs if os.path.isdir(d)]

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        info = {}
        opt = self.get_option_single('atlas_libs', 'libraries')
        atlas_libs = self.get_libs(opt, self._lib_names + self._lib_atlas)
        lapack_libs = self.get_libs('lapack_libs', self._lib_lapack)
        atlas = None
        lapack = None
        atlas_1 = None
        for d in lib_dirs:
            atlas = self.check_libs2(d, atlas_libs, [])
            if atlas is not None:
                lib_dirs2 = [d] + self.combine_paths(d, ['atlas*', 'ATLAS*'])
                lapack = self.check_libs2(lib_dirs2, lapack_libs, [])
                if lapack is not None:
                    break
            if atlas:
                atlas_1 = atlas
        log.info(self.__class__)
        if atlas is None:
            atlas = atlas_1
        if atlas is None:
            return
        include_dirs = self.get_include_dirs()
        h = self.combine_paths(lib_dirs + include_dirs, 'cblas.h') or [None]
        h = h[0]
        if h:
            h = os.path.dirname(h)
            dict_append(info, include_dirs=[h])
        info['language'] = 'c'
        if lapack is not None:
            dict_append(info, **lapack)
            dict_append(info, **atlas)
        elif 'lapack_atlas' in atlas['libraries']:
            dict_append(info, **atlas)
            dict_append(info, define_macros=[('ATLAS_WITH_LAPACK_ATLAS', None)])
            self.set_info(**info)
            return
        else:
            dict_append(info, **atlas)
            dict_append(info, define_macros=[('ATLAS_WITHOUT_LAPACK', None)])
            message = textwrap.dedent('\n                *********************************************************************\n                    Could not find lapack library within the ATLAS installation.\n                *********************************************************************\n                ')
            warnings.warn(message, stacklevel=2)
            self.set_info(**info)
            return
        lapack_dir = lapack['library_dirs'][0]
        lapack_name = lapack['libraries'][0]
        lapack_lib = None
        lib_prefixes = ['lib']
        if sys.platform == 'win32':
            lib_prefixes.append('')
        for e in self.library_extensions():
            for prefix in lib_prefixes:
                fn = os.path.join(lapack_dir, prefix + lapack_name + e)
                if os.path.exists(fn):
                    lapack_lib = fn
                    break
            if lapack_lib:
                break
        if lapack_lib is not None:
            sz = os.stat(lapack_lib)[6]
            if sz <= 4000 * 1024:
                message = textwrap.dedent('\n                    *********************************************************************\n                        Lapack library (from ATLAS) is probably incomplete:\n                          size of %s is %sk (expected >4000k)\n\n                        Follow the instructions in the KNOWN PROBLEMS section of the file\n                        numpy/INSTALL.txt.\n                    *********************************************************************\n                    ') % (lapack_lib, sz / 1024)
                warnings.warn(message, stacklevel=2)
            else:
                info['language'] = 'f77'
        atlas_version, atlas_extra_info = get_atlas_version(**atlas)
        dict_append(info, **atlas_extra_info)
        self.set_info(**info)