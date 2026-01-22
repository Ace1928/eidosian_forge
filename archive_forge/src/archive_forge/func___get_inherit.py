import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def __get_inherit(self):
    if self.value is None:
        raise ValueError('Handle is already closed!')
    return bool(GetHandleInformation(self.value) & HANDLE_FLAG_INHERIT)