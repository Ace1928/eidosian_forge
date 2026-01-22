import platform
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.ccompiler import simple_version_match
class IntelCCompilerW(MSVCCompiler):
    """
        A modified Intel compiler compatible with an MSVC-built Python.
        """
    compiler_type = 'intelw'
    compiler_cxx = 'icl'

    def __init__(self, verbose=0, dry_run=0, force=0):
        MSVCCompiler.__init__(self, verbose, dry_run, force)
        version_match = simple_version_match(start='Intel\\(R\\).*?32,')
        self.__version = version_match

    def initialize(self, plat_name=None):
        MSVCCompiler.initialize(self, plat_name)
        self.cc = self.find_exe('icl.exe')
        self.lib = self.find_exe('xilib')
        self.linker = self.find_exe('xilink')
        self.compile_options = ['/nologo', '/O3', '/MD', '/W3', '/Qstd=c99']
        self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/Qstd=c99', '/Z7', '/D_DEBUG']