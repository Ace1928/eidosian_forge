from numpy.distutils.fcompiler import FCompiler
class FujitsuFCompiler(FCompiler):
    compiler_type = 'fujitsu'
    description = 'Fujitsu Fortran Compiler'
    possible_executables = ['frt']
    version_pattern = 'frt \\(FRT\\) (?P<version>[a-z\\d.]+)'
    executables = {'version_cmd': ['<F77>', '--version'], 'compiler_f77': ['frt', '-Fixed'], 'compiler_fix': ['frt', '-Fixed'], 'compiler_f90': ['frt'], 'linker_so': ['frt', '-shared'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    pic_flags = ['-KPIC']
    module_dir_switch = '-M'
    module_include_switch = '-I'

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_debug(self):
        return ['-g']

    def runtime_library_dir_option(self, dir):
        return f'-Wl,-rpath={dir}'

    def get_libraries(self):
        return ['fj90f', 'fj90i', 'fjsrcinfo']