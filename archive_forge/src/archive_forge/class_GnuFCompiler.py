import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
class GnuFCompiler(FCompiler):
    compiler_type = 'gnu'
    compiler_aliases = ('g77',)
    description = 'GNU Fortran 77 compiler'

    def gnu_version_match(self, version_string):
        """Handle the different versions of GNU fortran compilers"""
        while version_string.startswith('gfortran: warning'):
            version_string = version_string[version_string.find('\n') + 1:].strip()
        if len(version_string) <= 20:
            m = re.search('([0-9.]+)', version_string)
            if m:
                if version_string.startswith('GNU Fortran'):
                    return ('g77', m.group(1))
                elif m.start() == 0:
                    return ('gfortran', m.group(1))
        else:
            m = re.search('GNU Fortran\\s+95.*?([0-9-.]+)', version_string)
            if m:
                return ('gfortran', m.group(1))
            m = re.search('GNU Fortran.*?\\-?([0-9-.]+\\.[0-9-.]+)', version_string)
            if m:
                v = m.group(1)
                if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                    return ('g77', v)
                else:
                    return ('gfortran', v)
        err = 'A valid Fortran version was not found in this string:\n'
        raise ValueError(err + version_string)

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'g77':
            return None
        return v[1]
    possible_executables = ['g77', 'f77']
    executables = {'version_cmd': [None, '-dumpversion'], 'compiler_f77': [None, '-g', '-Wall', '-fno-second-underscore'], 'compiler_f90': None, 'compiler_fix': None, 'linker_so': [None, '-g', '-Wall'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib'], 'linker_exe': [None, '-g', '-Wall']}
    module_dir_switch = None
    module_include_switch = None
    if os.name != 'nt' and sys.platform != 'cygwin':
        pic_flags = ['-fPIC']
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')
    g2c = 'g2c'
    suggested_f90_compiler = 'gnu95'

    def get_flags_linker_so(self):
        opt = self.linker_so[1:]
        if sys.platform == 'darwin':
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
            if not target:
                import sysconfig
                target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
                if not target:
                    target = '10.9'
                    s = f'Env. variable MACOSX_DEPLOYMENT_TARGET set to {target}'
                    warnings.warn(s, stacklevel=2)
                os.environ['MACOSX_DEPLOYMENT_TARGET'] = str(target)
            opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
        else:
            opt.append('-shared')
        if sys.platform.startswith('sunos'):
            opt.append('-mimpure-text')
        return opt

    def get_libgcc_dir(self):
        try:
            output = subprocess.check_output(self.compiler_f77 + ['-print-libgcc-file-name'])
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            return os.path.dirname(output)
        return None

    def get_libgfortran_dir(self):
        if sys.platform[:5] == 'linux':
            libgfortran_name = 'libgfortran.so'
        elif sys.platform == 'darwin':
            libgfortran_name = 'libgfortran.dylib'
        else:
            libgfortran_name = None
        libgfortran_dir = None
        if libgfortran_name:
            find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
            try:
                output = subprocess.check_output(self.compiler_f77 + find_lib_arg)
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                output = filepath_from_subprocess_output(output)
                libgfortran_dir = os.path.dirname(output)
        return libgfortran_dir

    def get_library_dirs(self):
        opt = []
        if sys.platform[:5] != 'linux':
            d = self.get_libgcc_dir()
            if d:
                if sys.platform == 'win32' and (not d.startswith('/usr/lib')):
                    d = os.path.normpath(d)
                    path = os.path.join(d, 'lib%s.a' % self.g2c)
                    if not os.path.exists(path):
                        root = os.path.join(d, *(os.pardir,) * 4)
                        d2 = os.path.abspath(os.path.join(root, 'lib'))
                        path = os.path.join(d2, 'lib%s.a' % self.g2c)
                        if os.path.exists(path):
                            opt.append(d2)
                opt.append(d)
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        opt = []
        d = self.get_libgcc_dir()
        if d is not None:
            g2c = self.g2c + '-pic'
            f = self.static_lib_format % (g2c, self.static_lib_extension)
            if not os.path.isfile(os.path.join(d, f)):
                g2c = self.g2c
        else:
            g2c = self.g2c
        if g2c is not None:
            opt.append(g2c)
        c_compiler = self.c_compiler
        if sys.platform == 'win32' and c_compiler and (c_compiler.compiler_type == 'msvc'):
            opt.append('gcc')
        if sys.platform == 'darwin':
            opt.append('cc_dynamic')
        return opt

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        v = self.get_version()
        if v and v <= '3.3.3':
            opt = ['-O2']
        else:
            opt = ['-O3']
        opt.append('-funroll-loops')
        return opt

    def _c_arch_flags(self):
        """ Return detected arch flags from CFLAGS """
        import sysconfig
        try:
            cflags = sysconfig.get_config_vars()['CFLAGS']
        except KeyError:
            return []
        arch_re = re.compile('-arch\\s+(\\w+)')
        arch_flags = []
        for arch in arch_re.findall(cflags):
            arch_flags += ['-arch', arch]
        return arch_flags

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            raise NotImplementedError
        assert ',' not in dir
        if sys.platform == 'darwin':
            return f'-Wl,-rpath,{dir}'
        elif sys.platform.startswith(('aix', 'os400')):
            return f'-Wl,-blibpath:{dir}'
        else:
            return f'-Wl,-rpath={dir}'