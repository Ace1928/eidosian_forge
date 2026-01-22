from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
class ExitProcessEvent(Event):
    """
    Process termination event.
    """
    eventMethod = 'exit_process'
    eventName = 'Process termination event'
    eventDescription = 'A process has finished executing.'

    def get_exit_code(self):
        """
        @rtype:  int
        @return: Exit code of the process.
        """
        return self.raw.u.ExitProcess.dwExitCode

    def get_filename(self):
        """
        @rtype:  None or str
        @return: Filename of the main module.
            C{None} if the filename is unknown.
        """
        return self.get_module().get_filename()

    def get_image_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_module_base()

    def get_module_base(self):
        """
        @rtype:  int
        @return: Base address of the main module.
        """
        return self.get_module().get_base()

    def get_module(self):
        """
        @rtype:  L{Module}
        @return: Main module of the process.
        """
        return self.get_process().get_main_module()