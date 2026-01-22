import sys
from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
class IntelItaniumVisualFCompiler(IntelVisualFCompiler):
    compiler_type = 'intelev'
    description = 'Intel Visual Fortran Compiler for Itanium apps'
    version_match = intel_version_match('Itanium')
    possible_executables = ['efl']
    ar_exe = IntelVisualFCompiler.ar_exe
    executables = {'version_cmd': None, 'compiler_f77': [None, '-FI', '-w90', '-w95'], 'compiler_fix': [None, '-FI', '-4L72', '-w'], 'compiler_f90': [None], 'linker_so': ['<F90>', '-shared'], 'archiver': [ar_exe, '/verbose', '/OUT:'], 'ranlib': None}