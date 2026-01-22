from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_AMD64
from winappdbg.win32 import context_i386
def Wow64GetThreadContext(hThread, ContextFlags=None):
    _Wow64GetThreadContext = windll.kernel32.Wow64GetThreadContext
    _Wow64GetThreadContext.argtypes = [HANDLE, PWOW64_CONTEXT]
    _Wow64GetThreadContext.restype = bool
    _Wow64GetThreadContext.errcheck = RaiseIfZero
    Context = WOW64_CONTEXT()
    if ContextFlags is None:
        Context.ContextFlags = WOW64_CONTEXT_ALL | WOW64_CONTEXT_i386
    else:
        Context.ContextFlags = ContextFlags
    _Wow64GetThreadContext(hThread, byref(Context))
    return Context.to_dict()