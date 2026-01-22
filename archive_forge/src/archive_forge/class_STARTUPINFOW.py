import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class STARTUPINFOW(Structure):
    _fields_ = [('cb', DWORD), ('lpReserved', LPWSTR), ('lpDesktop', LPWSTR), ('lpTitle', LPWSTR), ('dwX', DWORD), ('dwY', DWORD), ('dwXSize', DWORD), ('dwYSize', DWORD), ('dwXCountChars', DWORD), ('dwYCountChars', DWORD), ('dwFillAttribute', DWORD), ('dwFlags', DWORD), ('wShowWindow', WORD), ('cbReserved2', WORD), ('lpReserved2', LPVOID), ('hStdInput', HANDLE), ('hStdOutput', HANDLE), ('hStdError', HANDLE)]