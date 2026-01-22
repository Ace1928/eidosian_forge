from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class HardwareBreakpoint(Breakpoint):
    """
    Hardware breakpoint (using debug registers).

    @see: L{Debug.watch_variable}

    @group Information:
        get_slot, get_trigger, get_watch

    @group Trigger flags:
        BREAK_ON_EXECUTION, BREAK_ON_WRITE, BREAK_ON_ACCESS

    @group Watch size flags:
        WATCH_BYTE, WATCH_WORD, WATCH_DWORD, WATCH_QWORD

    @type BREAK_ON_EXECUTION: int
    @cvar BREAK_ON_EXECUTION: Break on execution.

    @type BREAK_ON_WRITE: int
    @cvar BREAK_ON_WRITE: Break on write.

    @type BREAK_ON_ACCESS: int
    @cvar BREAK_ON_ACCESS: Break on read or write.

    @type WATCH_BYTE: int
    @cvar WATCH_BYTE: Watch a byte.

    @type WATCH_WORD: int
    @cvar WATCH_WORD: Watch a word (2 bytes).

    @type WATCH_DWORD: int
    @cvar WATCH_DWORD: Watch a double word (4 bytes).

    @type WATCH_QWORD: int
    @cvar WATCH_QWORD: Watch one quad word (8 bytes).

    @type validTriggers: tuple
    @cvar validTriggers: Valid trigger flag values.

    @type validWatchSizes: tuple
    @cvar validWatchSizes: Valid watch flag values.
    """
    typeName = 'hardware breakpoint'
    BREAK_ON_EXECUTION = DebugRegister.BREAK_ON_EXECUTION
    BREAK_ON_WRITE = DebugRegister.BREAK_ON_WRITE
    BREAK_ON_ACCESS = DebugRegister.BREAK_ON_ACCESS
    WATCH_BYTE = DebugRegister.WATCH_BYTE
    WATCH_WORD = DebugRegister.WATCH_WORD
    WATCH_DWORD = DebugRegister.WATCH_DWORD
    WATCH_QWORD = DebugRegister.WATCH_QWORD
    validTriggers = (BREAK_ON_EXECUTION, BREAK_ON_WRITE, BREAK_ON_ACCESS)
    validWatchSizes = (WATCH_BYTE, WATCH_WORD, WATCH_DWORD, WATCH_QWORD)

    def __init__(self, address, triggerFlag=BREAK_ON_ACCESS, sizeFlag=WATCH_DWORD, condition=True, action=None):
        """
        Hardware breakpoint object.

        @see: L{Breakpoint.__init__}

        @type  address: int
        @param address: Memory address for breakpoint.

        @type  triggerFlag: int
        @param triggerFlag: Trigger of breakpoint. Must be one of the following:

             - L{BREAK_ON_EXECUTION}

               Break on code execution.

             - L{BREAK_ON_WRITE}

               Break on memory read or write.

             - L{BREAK_ON_ACCESS}

               Break on memory write.

        @type  sizeFlag: int
        @param sizeFlag: Size of breakpoint. Must be one of the following:

             - L{WATCH_BYTE}

               One (1) byte in size.

             - L{WATCH_WORD}

               Two (2) bytes in size.

             - L{WATCH_DWORD}

               Four (4) bytes in size.

             - L{WATCH_QWORD}

               Eight (8) bytes in size.

        @type  condition: function
        @param condition: (Optional) Condition callback function.

        @type  action: function
        @param action: (Optional) Action callback function.
        """
        if win32.arch not in (win32.ARCH_I386, win32.ARCH_AMD64):
            msg = 'Hardware breakpoints not supported for %s' % win32.arch
            raise NotImplementedError(msg)
        if sizeFlag == self.WATCH_BYTE:
            size = 1
        elif sizeFlag == self.WATCH_WORD:
            size = 2
        elif sizeFlag == self.WATCH_DWORD:
            size = 4
        elif sizeFlag == self.WATCH_QWORD:
            size = 8
        else:
            msg = 'Invalid size flag for hardware breakpoint (%s)'
            msg = msg % repr(sizeFlag)
            raise ValueError(msg)
        if triggerFlag not in self.validTriggers:
            msg = 'Invalid trigger flag for hardware breakpoint (%s)'
            msg = msg % repr(triggerFlag)
            raise ValueError(msg)
        Breakpoint.__init__(self, address, size, condition, action)
        self.__trigger = triggerFlag
        self.__watch = sizeFlag
        self.__slot = None

    def __clear_bp(self, aThread):
        """
        Clears this breakpoint from the debug registers.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
        if self.__slot is not None:
            aThread.suspend()
            try:
                ctx = aThread.get_context(win32.CONTEXT_DEBUG_REGISTERS)
                DebugRegister.clear_bp(ctx, self.__slot)
                aThread.set_context(ctx)
                self.__slot = None
            finally:
                aThread.resume()

    def __set_bp(self, aThread):
        """
        Sets this breakpoint in the debug registers.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
        if self.__slot is None:
            aThread.suspend()
            try:
                ctx = aThread.get_context(win32.CONTEXT_DEBUG_REGISTERS)
                self.__slot = DebugRegister.find_slot(ctx)
                if self.__slot is None:
                    msg = 'No available hardware breakpoint slots for thread ID %d'
                    msg = msg % aThread.get_tid()
                    raise RuntimeError(msg)
                DebugRegister.set_bp(ctx, self.__slot, self.get_address(), self.__trigger, self.__watch)
                aThread.set_context(ctx)
            finally:
                aThread.resume()

    def get_slot(self):
        """
        @rtype:  int
        @return: The debug register number used by this breakpoint,
            or C{None} if the breakpoint is not active.
        """
        return self.__slot

    def get_trigger(self):
        """
        @see: L{validTriggers}
        @rtype:  int
        @return: The breakpoint trigger flag.
        """
        return self.__trigger

    def get_watch(self):
        """
        @see: L{validWatchSizes}
        @rtype:  int
        @return: The breakpoint watch flag.
        """
        return self.__watch

    def disable(self, aProcess, aThread):
        if not self.is_disabled():
            self.__clear_bp(aThread)
        super(HardwareBreakpoint, self).disable(aProcess, aThread)

    def enable(self, aProcess, aThread):
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aThread)
        super(HardwareBreakpoint, self).enable(aProcess, aThread)

    def one_shot(self, aProcess, aThread):
        if not self.is_enabled() and (not self.is_one_shot()):
            self.__set_bp(aThread)
        super(HardwareBreakpoint, self).one_shot(aProcess, aThread)

    def running(self, aProcess, aThread):
        self.__clear_bp(aThread)
        super(HardwareBreakpoint, self).running(aProcess, aThread)
        aThread.set_tf()