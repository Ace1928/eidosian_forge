from winappdbg.win32.defines import *
def _get_os(osvi=None):
    """
    Determines the current operating system.

    This function allows you to quickly tell apart major OS differences.
    For more detailed information call L{GetVersionEx} instead.

    @note:
        Wine reports itself as Windows XP 32 bits
        (even if the Linux host is 64 bits).
        ReactOS may report itself as Windows 2000 or Windows XP,
        depending on the version of ReactOS.

    @type  osvi: L{OSVERSIONINFOEXA}
    @param osvi: Optional. The return value from L{GetVersionEx}.

    @rtype: str
    @return:
        One of the following values:
         - L{OS_UNKNOWN} (C{"Unknown"})
         - L{OS_NT} (C{"Windows NT"})
         - L{OS_W2K} (C{"Windows 2000"})
         - L{OS_XP} (C{"Windows XP"})
         - L{OS_XP_64} (C{"Windows XP (64 bits)"})
         - L{OS_W2K3} (C{"Windows 2003"})
         - L{OS_W2K3_64} (C{"Windows 2003 (64 bits)"})
         - L{OS_W2K3R2} (C{"Windows 2003 R2"})
         - L{OS_W2K3R2_64} (C{"Windows 2003 R2 (64 bits)"})
         - L{OS_W2K8} (C{"Windows 2008"})
         - L{OS_W2K8_64} (C{"Windows 2008 (64 bits)"})
         - L{OS_W2K8R2} (C{"Windows 2008 R2"})
         - L{OS_W2K8R2_64} (C{"Windows 2008 R2 (64 bits)"})
         - L{OS_VISTA} (C{"Windows Vista"})
         - L{OS_VISTA_64} (C{"Windows Vista (64 bits)"})
         - L{OS_W7} (C{"Windows 7"})
         - L{OS_W7_64} (C{"Windows 7 (64 bits)"})
    """
    if not osvi:
        osvi = GetVersionEx()
    if osvi.dwPlatformId == VER_PLATFORM_WIN32_NT and osvi.dwMajorVersion > 4:
        if osvi.dwMajorVersion == 6:
            if osvi.dwMinorVersion == 0:
                if osvi.wProductType == VER_NT_WORKSTATION:
                    if bits == 64 or wow64:
                        return 'Windows Vista (64 bits)'
                    return 'Windows Vista'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2008 (64 bits)'
                    return 'Windows 2008'
            if osvi.dwMinorVersion == 1:
                if osvi.wProductType == VER_NT_WORKSTATION:
                    if bits == 64 or wow64:
                        return 'Windows 7 (64 bits)'
                    return 'Windows 7'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2008 R2 (64 bits)'
                    return 'Windows 2008 R2'
        if osvi.dwMajorVersion == 5:
            if osvi.dwMinorVersion == 2:
                if GetSystemMetrics(SM_SERVERR2):
                    if bits == 64 or wow64:
                        return 'Windows 2003 R2 (64 bits)'
                    return 'Windows 2003 R2'
                if osvi.wSuiteMask in (VER_SUITE_STORAGE_SERVER, VER_SUITE_WH_SERVER):
                    if bits == 64 or wow64:
                        return 'Windows 2003 (64 bits)'
                    return 'Windows 2003'
                if osvi.wProductType == VER_NT_WORKSTATION and arch == ARCH_AMD64:
                    return 'Windows XP (64 bits)'
                else:
                    if bits == 64 or wow64:
                        return 'Windows 2003 (64 bits)'
                    return 'Windows 2003'
            if osvi.dwMinorVersion == 1:
                return 'Windows XP'
            if osvi.dwMinorVersion == 0:
                return 'Windows 2000'
        if osvi.dwMajorVersion == 4:
            return 'Windows NT'
    return 'Unknown'