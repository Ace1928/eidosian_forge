import platform
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.ccompiler import simple_version_match
class IntelEM64TCCompilerW(IntelCCompilerW):
    """
        A modified Intel x86_64 compiler compatible with
        a 64bit MSVC-built Python.
        """
    compiler_type = 'intelemw'

    def __init__(self, verbose=0, dry_run=0, force=0):
        MSVCCompiler.__init__(self, verbose, dry_run, force)
        version_match = simple_version_match(start='Intel\\(R\\).*?64,')
        self.__version = version_match