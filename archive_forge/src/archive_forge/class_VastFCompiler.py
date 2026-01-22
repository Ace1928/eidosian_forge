import os
from numpy.distutils.fcompiler.gnu import GnuFCompiler
class VastFCompiler(GnuFCompiler):
    compiler_type = 'vast'
    compiler_aliases = ()
    description = 'Pacific-Sierra Research Fortran 90 Compiler'
    version_pattern = '\\s*Pacific-Sierra Research vf90 (Personal|Professional)\\s+(?P<version>[^\\s]*)'
    object_switch = ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile '
    executables = {'version_cmd': ['vf90', '-v'], 'compiler_f77': ['g77'], 'compiler_fix': ['f90', '-Wv,-ya'], 'compiler_f90': ['f90'], 'linker_so': ['<F90>'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    module_dir_switch = None
    module_include_switch = None

    def find_executables(self):
        pass

    def get_version_cmd(self):
        f90 = self.compiler_f90[0]
        d, b = os.path.split(f90)
        vf90 = os.path.join(d, 'v' + b)
        return vf90

    def get_flags_arch(self):
        vast_version = self.get_version()
        gnu = GnuFCompiler()
        gnu.customize(None)
        self.version = gnu.get_version()
        opt = GnuFCompiler.get_flags_arch(self)
        self.version = vast_version
        return opt