from winappdbg.win32.defines import *
class OSVERSIONINFOEXA(Structure):
    _fields_ = [('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', CHAR * 128), ('wServicePackMajor', WORD), ('wServicePackMinor', WORD), ('wSuiteMask', WORD), ('wProductType', BYTE), ('wReserved', BYTE)]