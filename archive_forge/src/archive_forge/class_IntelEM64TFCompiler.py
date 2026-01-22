import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelEM64TFCompiler(IntelFCompiler):
    compiler_type = 'intelem'
    compiler_aliases = ()
    description = 'Intel Fortran Compiler for 64-bit apps'
    version_match = intel_version_match('EM64T-based|Intel\\(R\\) 64|64|IA-64|64-bit')
    possible_executables = ['ifort', 'efort', 'efc']
    executables = {'version_cmd': None, 'compiler_f77': [None, '-FI'], 'compiler_fix': [None, '-FI'], 'compiler_f90': [None], 'linker_so': ['<F90>', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}