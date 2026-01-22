from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
def GetThreadContext(hThread, ContextFlags=None, raw=False):
    _GetThreadContext = windll.kernel32.GetThreadContext
    _GetThreadContext.argtypes = [HANDLE, LPCONTEXT]
    _GetThreadContext.restype = bool
    _GetThreadContext.errcheck = RaiseIfZero
    if ContextFlags is None:
        ContextFlags = CONTEXT_ALL | CONTEXT_i386
    Context = CONTEXT()
    Context.ContextFlags = ContextFlags
    _GetThreadContext(hThread, byref(Context))
    if raw:
        return Context
    return Context.to_dict()