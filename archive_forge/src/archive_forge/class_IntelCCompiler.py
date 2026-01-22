import platform
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.ccompiler import simple_version_match
class IntelCCompiler(UnixCCompiler):
    """A modified Intel compiler compatible with a GCC-built Python."""
    compiler_type = 'intel'
    cc_exe = 'icc'
    cc_args = 'fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        self.cc_exe = 'icc -fPIC -fp-model strict -O3 -fomit-frame-pointer -{}'.format(mpopt)
        compiler = self.cc_exe
        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        self.set_executables(compiler=compiler, compiler_so=compiler, compiler_cxx=compiler, archiver='xiar' + ' cru', linker_exe=compiler + ' -shared-intel', linker_so=compiler + ' ' + shared_flag + ' -shared-intel')