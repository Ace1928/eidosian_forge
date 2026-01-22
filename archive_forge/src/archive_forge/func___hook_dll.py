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
def __hook_dll(self, event):
    """
        Hook the requested API calls (in self.apiHooks).

        This method is called automatically whenever a DLL is loaded.
        """
    debug = event.debug
    pid = event.get_pid()
    for hook_api_stub in self.__get_hooks_for_dll(event):
        hook_api_stub.hook(debug, pid)