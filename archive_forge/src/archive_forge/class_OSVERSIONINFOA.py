from winappdbg.win32.defines import *
class OSVERSIONINFOA(Structure):
    _fields_ = [('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', CHAR * 128)]