from winappdbg.win32.defines import *
class OSVERSIONINFOEXW(Structure):
    _fields_ = [('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', WCHAR * 128), ('wServicePackMajor', WORD), ('wServicePackMinor', WORD), ('wSuiteMask', WORD), ('wProductType', BYTE), ('wReserved', BYTE)]