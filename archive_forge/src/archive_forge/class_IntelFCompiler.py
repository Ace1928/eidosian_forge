import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelFCompiler(BaseIntelFCompiler):
    compiler_type = 'intel'
    compiler_aliases = ('ifort',)
    description = 'Intel Fortran Compiler for 32-bit apps'
    version_match = intel_version_match('32-bit|IA-32')
    possible_executables = ['ifort', 'ifc']
    executables = {'version_cmd': None, 'compiler_f77': [None, '-72', '-w90', '-w95'], 'compiler_f90': [None], 'compiler_fix': [None, '-FI'], 'linker_so': ['<F90>', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    pic_flags = ['-fPIC']
    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags_free(self):
        return ['-FR']

    def get_flags(self):
        return ['-fPIC']

    def get_flags_opt(self):
        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        return ['-fp-model', 'strict', '-O1', '-assume', 'minus0', '-{}'.format(mpopt)]

    def get_flags_arch(self):
        return []

    def get_flags_linker_so(self):
        opt = FCompiler.get_flags_linker_so(self)
        v = self.get_version()
        if v and v >= '8.0':
            opt.append('-nofor_main')
        if sys.platform == 'darwin':
            try:
                idx = opt.index('-shared')
                opt.remove('-shared')
            except ValueError:
                idx = 0
            opt[idx:idx] = ['-dynamiclib', '-Wl,-undefined,dynamic_lookup']
        return opt