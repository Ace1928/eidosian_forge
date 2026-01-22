from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def CommandLineToArgvW(lpCmdLine):
    _CommandLineToArgvW = windll.shell32.CommandLineToArgvW
    _CommandLineToArgvW.argtypes = [LPVOID, POINTER(ctypes.c_int)]
    _CommandLineToArgvW.restype = LPVOID
    if not lpCmdLine:
        lpCmdLine = None
    argc = ctypes.c_int(0)
    vptr = ctypes.windll.shell32.CommandLineToArgvW(lpCmdLine, byref(argc))
    if vptr == NULL:
        raise ctypes.WinError()
    argv = vptr
    try:
        argc = argc.value
        if argc <= 0:
            raise ctypes.WinError()
        argv = ctypes.cast(argv, ctypes.POINTER(LPWSTR * argc))
        argv = [argv.contents[i] for i in compat.xrange(0, argc)]
    finally:
        if vptr is not None:
            LocalFree(vptr)
    return argv