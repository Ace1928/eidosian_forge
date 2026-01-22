from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class _Hook_i386(Hook):
    """
    Implementation details for L{Hook} on the L{win32.ARCH_I386} architecture.
    """
    __new__ = object.__new__

    def _calc_signature(self, signature):
        self._cast_signature_pointers_to_void(signature)

        class Arguments(ctypes.Structure):
            _fields_ = [('arg_%s' % i, signature[i]) for i in compat.xrange(len(signature) - 1, -1, -1)]
        return Arguments

    def _get_return_address(self, aProcess, aThread):
        return aProcess.read_pointer(aThread.get_sp())

    def _get_function_arguments(self, aProcess, aThread):
        if self._signature:
            params = aThread.read_stack_structure(self._signature, offset=win32.sizeof(win32.LPVOID))
        elif self._paramCount:
            params = aThread.read_stack_dwords(self._paramCount, offset=win32.sizeof(win32.LPVOID))
        else:
            params = ()
        return params

    def _get_return_value(self, aThread):
        ctx = aThread.get_context(win32.CONTEXT_INTEGER)
        return ctx['Eax']