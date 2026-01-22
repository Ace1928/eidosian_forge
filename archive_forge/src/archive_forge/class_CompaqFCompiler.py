import os
import sys
from numpy.distutils.fcompiler import FCompiler
from distutils.errors import DistutilsPlatformError
class CompaqFCompiler(FCompiler):
    compiler_type = 'compaq'
    description = 'Compaq Fortran Compiler'
    version_pattern = 'Compaq Fortran (?P<version>[^\\s]*).*'
    if sys.platform[:5] == 'linux':
        fc_exe = 'fort'
    else:
        fc_exe = 'f90'
    executables = {'version_cmd': ['<F90>', '-version'], 'compiler_f77': [fc_exe, '-f77rtl', '-fixed'], 'compiler_fix': [fc_exe, '-fixed'], 'compiler_f90': [fc_exe], 'linker_so': ['<F90>'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        return ['-assume no2underscore', '-nomixed_str_len_arg']

    def get_flags_debug(self):
        return ['-g', '-check bounds']

    def get_flags_opt(self):
        return ['-O4', '-align dcommons', '-assume bigarrays', '-assume nozsize', '-math_library fast']

    def get_flags_arch(self):
        return ['-arch host', '-tune host']

    def get_flags_linker_so(self):
        if sys.platform[:5] == 'linux':
            return ['-shared']
        return ['-shared', '-Wl,-expect_unresolved,*']