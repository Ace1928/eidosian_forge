from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _get_function_arguments(self, aProcess, aThread):
    if self._signature:
        args_count, RegisterArguments, FloatArguments, StackArguments = self._signature
        arguments = {}
        if StackArguments:
            address = aThread.get_sp() + win32.sizeof(win32.LPVOID)
            stack_struct = aProcess.read_structure(address, StackArguments)
            stack_args = dict([(name, stack_struct.__getattribute__(name)) for name, type in stack_struct._fields_])
            arguments.update(stack_args)
        flags = 0
        if RegisterArguments:
            flags = flags | win32.CONTEXT_INTEGER
        if FloatArguments:
            flags = flags | win32.CONTEXT_MMX_REGISTERS
        if flags:
            ctx = aThread.get_context(flags)
            if RegisterArguments:
                buffer = (win32.QWORD * 4)(ctx['Rcx'], ctx['Rdx'], ctx['R8'], ctx['R9'])
                reg_args = self._get_arguments_from_buffer(buffer, RegisterArguments)
                arguments.update(reg_args)
            if FloatArguments:
                buffer = (win32.M128A * 4)(ctx['XMM0'], ctx['XMM1'], ctx['XMM2'], ctx['XMM3'])
                float_args = self._get_arguments_from_buffer(buffer, FloatArguments)
                arguments.update(float_args)
        params = tuple([arguments['arg_%d' % i] for i in compat.xrange(args_count)])
    else:
        params = ()
    return params