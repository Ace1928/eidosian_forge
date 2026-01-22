from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@staticmethod
def adjust_privileges(state, privileges):
    """
        Requests or drops privileges.

        @type  state: bool
        @param state: C{True} to request, C{False} to drop.

        @type  privileges: list(int)
        @param privileges: Privileges to request or drop.

        @raise WindowsError: Raises an exception on error.
        """
    with win32.OpenProcessToken(win32.GetCurrentProcess(), win32.TOKEN_ADJUST_PRIVILEGES) as hToken:
        NewState = ((priv, state) for priv in privileges)
        win32.AdjustTokenPrivileges(hToken, NewState)