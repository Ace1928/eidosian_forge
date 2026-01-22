from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class Engine(object):
    """
    Base class for disassembly engine adaptors.

    @type name: str
    @cvar name: Engine name to use with the L{Disassembler} class.

    @type desc: str
    @cvar desc: User friendly name of the disassembler engine.

    @type url: str
    @cvar url: Download URL.

    @type supported: set(str)
    @cvar supported: Set of supported processor architectures.
        For more details see L{win32.version._get_arch}.

    @type arch: str
    @ivar arch: Name of the processor architecture.
    """
    name = '<insert engine name here>'
    desc = '<insert engine description here>'
    url = '<insert download url here>'
    supported = set()

    def __init__(self, arch=None):
        """
        @type  arch: str
        @param arch: Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @raise NotImplementedError: This disassembler doesn't support the
            requested processor architecture.
        """
        self.arch = self._validate_arch(arch)
        try:
            self._import_dependencies()
        except ImportError:
            msg = "%s is not installed or can't be found. Download it from: %s"
            msg = msg % (self.name, self.url)
            raise NotImplementedError(msg)

    def _validate_arch(self, arch=None):
        """
        @type  arch: str
        @param arch: Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @rtype:  str
        @return: Name of the processor architecture.
            If not provided the current processor architecture is assumed.
            For more details see L{win32.version._get_arch}.

        @raise NotImplementedError: This disassembler doesn't support the
            requested processor architecture.
        """
        if not arch:
            arch = win32.arch
        if arch not in self.supported:
            msg = 'The %s engine cannot decode %s code.'
            msg = msg % (self.name, arch)
            raise NotImplementedError(msg)
        return arch

    def _import_dependencies(self):
        """
        Loads the dependencies for this disassembler.

        @raise ImportError: This disassembler cannot find or load the
            necessary dependencies to make it work.
        """
        raise SyntaxError('Subclasses MUST implement this method!')

    def decode(self, address, code):
        """
        @type  address: int
        @param address: Memory address where the code was read from.

        @type  code: str
        @param code: Machine code to disassemble.

        @rtype:  list of tuple( long, int, str, str )
        @return: List of tuples. Each tuple represents an assembly instruction
            and contains:
             - Memory address of instruction.
             - Size of instruction in bytes.
             - Disassembly line of instruction.
             - Hexadecimal dump of instruction.

        @raise NotImplementedError: This disassembler could not be loaded.
            This may be due to missing dependencies.
        """
        raise NotImplementedError()