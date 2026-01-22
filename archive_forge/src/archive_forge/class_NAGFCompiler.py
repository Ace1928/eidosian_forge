import sys
import re
from numpy.distutils.fcompiler import FCompiler
class NAGFCompiler(BaseNAGFCompiler):
    compiler_type = 'nag'
    description = 'NAGWare Fortran 95 Compiler'
    executables = {'version_cmd': ['<F90>', '-V'], 'compiler_f77': ['f95', '-fixed'], 'compiler_fix': ['f95', '-fixed'], 'compiler_f90': ['f95'], 'linker_so': ['<F90>'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedf95', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)

    def get_flags_arch(self):
        version = self.get_version()
        if version and version < '5.1':
            return ['-target=native']
        else:
            return BaseNAGFCompiler.get_flags_arch(self)

    def get_flags_debug(self):
        return ['-g', '-gline', '-g90', '-nan', '-C']