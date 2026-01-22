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
class flame_info(system_info):
    """ Usage of libflame for LAPACK operations

    This requires libflame to be compiled with lapack wrappers:

    ./configure --enable-lapack2flame ...

    Be aware that libflame 5.1.0 has some missing names in the shared library, so
    if you have problems, try the static flame library.
    """
    section = 'flame'
    _lib_names = ['flame']
    notfounderror = FlameNotFoundError

    def check_embedded_lapack(self, info):
        """ libflame does not necessarily have a wrapper for fortran LAPACK, we need to check """
        c = customized_ccompiler()
        tmpdir = tempfile.mkdtemp()
        s = textwrap.dedent('            void zungqr_();\n            int main(int argc, const char *argv[])\n            {\n                zungqr_();\n                return 0;\n            }')
        src = os.path.join(tmpdir, 'source.c')
        out = os.path.join(tmpdir, 'a.out')
        extra_args = info.get('extra_link_args', [])
        try:
            with open(src, 'w') as f:
                f.write(s)
            obj = c.compile([src], output_dir=tmpdir)
            try:
                c.link_executable(obj, out, libraries=info['libraries'], library_dirs=info['library_dirs'], extra_postargs=extra_args)
                return True
            except distutils.ccompiler.LinkError:
                return False
        finally:
            shutil.rmtree(tmpdir)

    def calc_info(self):
        lib_dirs = self.get_lib_dirs()
        flame_libs = self.get_libs('libraries', self._lib_names)
        info = self.check_libs2(lib_dirs, flame_libs, [])
        if info is None:
            return
        extra_info = self.calc_extra_info()
        dict_append(info, **extra_info)
        if self.check_embedded_lapack(info):
            self.set_info(**info)
        else:
            blas_info = get_info('blas_opt')
            if not blas_info:
                return
            for key in blas_info:
                if isinstance(blas_info[key], list):
                    info[key] = info.get(key, []) + blas_info[key]
                elif isinstance(blas_info[key], tuple):
                    info[key] = info.get(key, ()) + blas_info[key]
                else:
                    info[key] = info.get(key, '') + blas_info[key]
            if self.check_embedded_lapack(info):
                self.set_info(**info)