from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
def __restore_mbi(self, hProcess, new_mbi, old_mbi, bSkipMappedFiles, bSkipOnError):
    """
        Used internally by L{restore_memory_snapshot}.
        """
    try:
        if new_mbi.State != old_mbi.State:
            if new_mbi.is_free():
                if old_mbi.is_reserved():
                    address = win32.VirtualAllocEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_RESERVE, old_mbi.Protect)
                    if address != old_mbi.BaseAddress:
                        self.free(address)
                        msg = 'Error restoring region at address %s'
                        msg = msg % HexDump(old_mbi.BaseAddress, self.get_bits())
                        raise RuntimeError(msg)
                    new_mbi.Protect = old_mbi.Protect
                else:
                    address = win32.VirtualAllocEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_RESERVE | win32.MEM_COMMIT, old_mbi.Protect)
                    if address != old_mbi.BaseAddress:
                        self.free(address)
                        msg = 'Error restoring region at address %s'
                        msg = msg % HexDump(old_mbi.BaseAddress, self.get_bits())
                        raise RuntimeError(msg)
                    new_mbi.Protect = old_mbi.Protect
            elif new_mbi.is_reserved():
                if old_mbi.is_commited():
                    address = win32.VirtualAllocEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_COMMIT, old_mbi.Protect)
                    if address != old_mbi.BaseAddress:
                        self.free(address)
                        msg = 'Error restoring region at address %s'
                        msg = msg % HexDump(old_mbi.BaseAddress, self.get_bits())
                        raise RuntimeError(msg)
                    new_mbi.Protect = old_mbi.Protect
                else:
                    win32.VirtualFreeEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_RELEASE)
            elif old_mbi.is_reserved():
                win32.VirtualFreeEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_DECOMMIT)
            else:
                win32.VirtualFreeEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, win32.MEM_DECOMMIT | win32.MEM_RELEASE)
        new_mbi.State = old_mbi.State
        if old_mbi.is_commited() and old_mbi.Protect != new_mbi.Protect:
            win32.VirtualProtectEx(hProcess, old_mbi.BaseAddress, old_mbi.RegionSize, old_mbi.Protect)
            new_mbi.Protect = old_mbi.Protect
        if old_mbi.has_content():
            if old_mbi.Type != 0:
                if not bSkipMappedFiles:
                    self.poke(old_mbi.BaseAddress, old_mbi.content)
            else:
                self.write(old_mbi.BaseAddress, old_mbi.content)
            new_mbi.content = old_mbi.content
    except Exception:
        if not bSkipOnError:
            raise
        msg = 'Error restoring region at address %s: %s'
        msg = msg % (HexDump(old_mbi.BaseAddress, self.get_bits()), traceback.format_exc())
        warnings.warn(msg, RuntimeWarning)