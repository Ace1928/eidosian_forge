import os
import subprocess
from glob import glob
from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
from distutils.file_util import copy_file
from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt
def _libs_with_msvc_and_fortran(self, fcompiler, c_libraries, c_library_dirs):
    if fcompiler is None:
        return
    for libname in c_libraries:
        if libname.startswith('msvc'):
            continue
        fileexists = False
        for libdir in c_library_dirs or []:
            libfile = os.path.join(libdir, '%s.lib' % libname)
            if os.path.isfile(libfile):
                fileexists = True
                break
        if fileexists:
            continue
        fileexists = False
        for libdir in c_library_dirs:
            libfile = os.path.join(libdir, 'lib%s.a' % libname)
            if os.path.isfile(libfile):
                libfile2 = os.path.join(self.build_temp, libname + '.lib')
                copy_file(libfile, libfile2)
                if self.build_temp not in c_library_dirs:
                    c_library_dirs.append(self.build_temp)
                fileexists = True
                break
        if fileexists:
            continue
        log.warn('could not find library %r in directories %s' % (libname, c_library_dirs))
    f_lib_dirs = []
    for dir in fcompiler.library_dirs:
        if dir.startswith('/usr/lib'):
            try:
                dir = subprocess.check_output(['cygpath', '-w', dir])
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                dir = filepath_from_subprocess_output(dir)
        f_lib_dirs.append(dir)
    c_library_dirs.extend(f_lib_dirs)
    for lib in fcompiler.libraries:
        if not lib.startswith('msvc'):
            c_libraries.append(lib)
            p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
            if p:
                dst_name = os.path.join(self.build_temp, lib + '.lib')
                if not os.path.isfile(dst_name):
                    copy_file(p[0], dst_name)
                if self.build_temp not in c_library_dirs:
                    c_library_dirs.append(self.build_temp)