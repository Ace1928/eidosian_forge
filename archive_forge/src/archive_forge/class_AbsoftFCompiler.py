import os
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32
class AbsoftFCompiler(FCompiler):
    compiler_type = 'absoft'
    description = 'Absoft Corp Fortran Compiler'
    version_pattern = '(f90:.*?(Absoft Pro FORTRAN Version|FORTRAN 77 Compiler|Absoft Fortran Compiler Version|Copyright Absoft Corporation.*?Version))' + ' (?P<version>[^\\s*,]*)(.*?Absoft Corp|)'
    executables = {'version_cmd': None, 'compiler_f77': ['f77'], 'compiler_fix': ['f90'], 'compiler_f90': ['f90'], 'linker_so': ['<F90>'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    if os.name == 'nt':
        library_switch = '/out:'
    module_dir_switch = None
    module_include_switch = '-p'

    def update_executables(self):
        f = cyg2win32(dummy_fortran_file())
        self.executables['version_cmd'] = ['<F90>', '-V', '-c', f + '.f', '-o', f + '.o']

    def get_flags_linker_so(self):
        if os.name == 'nt':
            opt = ['/dll']
        elif self.get_version() >= '9.0':
            opt = ['-shared']
        else:
            opt = ['-K', 'shared']
        return opt

    def library_dir_option(self, dir):
        if os.name == 'nt':
            return ['-link', '/PATH:%s' % dir]
        return '-L' + dir

    def library_option(self, lib):
        if os.name == 'nt':
            return '%s.lib' % lib
        return '-l' + lib

    def get_library_dirs(self):
        opt = FCompiler.get_library_dirs(self)
        d = os.environ.get('ABSOFT')
        if d:
            if self.get_version() >= '10.0':
                prefix = 'sh'
            else:
                prefix = ''
            if cpu.is_64bit():
                suffix = '64'
            else:
                suffix = ''
            opt.append(os.path.join(d, '%slib%s' % (prefix, suffix)))
        return opt

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        if self.get_version() >= '11.0':
            opt.extend(['af90math', 'afio', 'af77math', 'amisc'])
        elif self.get_version() >= '10.0':
            opt.extend(['af90math', 'afio', 'af77math', 'U77'])
        elif self.get_version() >= '8.0':
            opt.extend(['f90math', 'fio', 'f77math', 'U77'])
        else:
            opt.extend(['fio', 'f90math', 'fmath', 'U77'])
        if os.name == 'nt':
            opt.append('COMDLG32')
        return opt

    def get_flags(self):
        opt = FCompiler.get_flags(self)
        if os.name != 'nt':
            opt.extend(['-s'])
            if self.get_version():
                if self.get_version() >= '8.2':
                    opt.append('-fpic')
        return opt

    def get_flags_f77(self):
        opt = FCompiler.get_flags_f77(self)
        opt.extend(['-N22', '-N90', '-N110'])
        v = self.get_version()
        if os.name == 'nt':
            if v and v >= '8.0':
                opt.extend(['-f', '-N15'])
        else:
            opt.append('-f')
            if v:
                if v <= '4.6':
                    opt.append('-B108')
                else:
                    opt.append('-N15')
        return opt

    def get_flags_f90(self):
        opt = FCompiler.get_flags_f90(self)
        opt.extend(['-YCFRL=1', '-YCOM_NAMES=LCS', '-YCOM_PFX', '-YEXT_PFX', '-YCOM_SFX=_', '-YEXT_SFX=_', '-YEXT_NAMES=LCS'])
        if self.get_version():
            if self.get_version() > '4.6':
                opt.extend(['-YDEALLOC=ALL'])
        return opt

    def get_flags_fix(self):
        opt = FCompiler.get_flags_fix(self)
        opt.extend(['-YCFRL=1', '-YCOM_NAMES=LCS', '-YCOM_PFX', '-YEXT_PFX', '-YCOM_SFX=_', '-YEXT_SFX=_', '-YEXT_NAMES=LCS'])
        opt.extend(['-f', 'fixed'])
        return opt

    def get_flags_opt(self):
        opt = ['-O']
        return opt