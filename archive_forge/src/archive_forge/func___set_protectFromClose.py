import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def __set_protectFromClose(self, value):
    if self.value is None:
        raise ValueError('Handle is already closed!')
    flag = (0, HANDLE_FLAG_PROTECT_FROM_CLOSE)[bool(value)]
    SetHandleInformation(self.value, flag, flag)