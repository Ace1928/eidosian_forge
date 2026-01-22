from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class NT_TIB(Structure):
    _fields_ = [('ExceptionList', PVOID), ('StackBase', PVOID), ('StackLimit', PVOID), ('SubSystemTib', PVOID), ('u', _NT_TIB_UNION), ('ArbitraryUserPointer', PVOID), ('Self', PVOID)]

    def __get_FiberData(self):
        return self.u.FiberData

    def __set_FiberData(self, value):
        self.u.FiberData = value
    FiberData = property(__get_FiberData, __set_FiberData)

    def __get_Version(self):
        return self.u.Version

    def __set_Version(self, value):
        self.u.Version = value
    Version = property(__get_Version, __set_Version)