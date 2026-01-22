from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class CodeBreakpoint(Breakpoint):
    """
    Code execution breakpoints (using an int3 opcode).

    @see: L{Debug.break_at}

    @type bpInstruction: str
    @cvar bpInstruction: Breakpoint instruction for the current processor.
    """
    typeName = 'code breakpoint'
    if win32.arch in (win32.ARCH_I386, win32.ARCH_AMD64):
        bpInstruction = 'Ì'

    def __init__(self, address, condition=True, action=None):
        """
        Code breakpoint object.

        @see: L{Breakpoint.__init__}

        @type  address: int
        @param address: Memory address for breakpoint.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

        @type  action: function
        @param action: (Optional) Action callback function.
        """
        if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            msg = 'Code breakpoints not supported for %s' % win32.arch
            raise NotImplementedError(msg)
        Breakpoint.__init__(self, address, len(self.bpInstruction), condition, action)
        self.__previousValue = self.bpInstruction

    def __set_bp(self, aProcess):
        """
        Writes a breakpoint instruction at the target address.

        @type  aProcess: L{Process}
        @param aProcess: Process object.
        """
        address = self.get_address()
        self.__previousValue = aProcess.read(address, len(self.bpInstruction))
        if self.__previousValue == self.bpInstruction:
            msg = 'Possible overlapping code breakpoints at %s'
            msg = msg % HexDump.address(address)
            warnings.warn(msg, BreakpointWarning)
        aProcess.write(address, self.bpInstruction)

    def __clear_bp(self, aProcess):
        """
        Restores the original byte at the target address.

        @type  aProcess: L{Process}
        @param aProcess: Process object.
        """
        address = self.get_address()
        currentValue = aProcess.read(address, len(self.bpInstruction))
        if currentValue == self.bpInstruction:
            aProcess.write(self.get_address(), self.__previousValue)
        else:
            self.__previousValue = currentValue
            msg = 'Overwritten code breakpoint at %s'
            msg = msg % HexDump.address(address)
            warnings.warn(msg, BreakpointWarning)

    def disable(self, aProcess, aThread):
        if not self.is_disabled() and (not self.is_running()):
            self.__clear_bp(aProcess)
        super(CodeBreakpoint, self).disable(aProcess, aThread)

    def enable(self, aProcess, aThread):
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aProcess)
        super(CodeBreakpoint, self).enable(aProcess, aThread)

    def one_shot(self, aProcess, aThread):
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aProcess)
        super(CodeBreakpoint, self).one_shot(aProcess, aThread)

    def running(self, aProcess, aThread):
        if self.is_enabled():
            self.__clear_bp(aProcess)
            aThread.set_tf()
        super(CodeBreakpoint, self).running(aProcess, aThread)