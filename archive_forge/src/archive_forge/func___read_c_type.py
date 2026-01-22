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
def __read_c_type(self, address, format, c_type):
    size = ctypes.sizeof(c_type)
    packed = self.read(address, size)
    if len(packed) != size:
        raise ctypes.WinError()
    return struct.unpack(format, packed)[0]