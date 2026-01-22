from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
class SHELLEXECUTEINFO(Structure):
    _fields_ = [('cbSize', DWORD), ('fMask', ULONG), ('hwnd', HWND), ('lpVerb', LPSTR), ('lpFile', LPSTR), ('lpParameters', LPSTR), ('lpDirectory', LPSTR), ('nShow', ctypes.c_int), ('hInstApp', HINSTANCE), ('lpIDList', LPVOID), ('lpClass', LPSTR), ('hkeyClass', HKEY), ('dwHotKey', DWORD), ('hIcon', HANDLE), ('hProcess', HANDLE)]

    def __get_hMonitor(self):
        return self.hIcon

    def __set_hMonitor(self, hMonitor):
        self.hIcon = hMonitor
    hMonitor = property(__get_hMonitor, __set_hMonitor)