from winappdbg.win32.defines import *
def _get_ntddi(osvi):
    """
    Determines the current operating system.

    This function allows you to quickly tell apart major OS differences.
    For more detailed information call L{kernel32.GetVersionEx} instead.

    @note:
        Wine reports itself as Windows XP 32 bits
        (even if the Linux host is 64 bits).
        ReactOS may report itself as Windows 2000 or Windows XP,
        depending on the version of ReactOS.

    @type  osvi: L{OSVERSIONINFOEXA}
    @param osvi: Optional. The return value from L{kernel32.GetVersionEx}.

    @rtype:  int
    @return: NTDDI version number.
    """
    if not osvi:
        osvi = GetVersionEx()
    ntddi = 0
    ntddi += (osvi.dwMajorVersion & 255) << 24
    ntddi += (osvi.dwMinorVersion & 255) << 16
    ntddi += (osvi.wServicePackMajor & 255) << 8
    ntddi += osvi.wServicePackMinor & 255
    return ntddi