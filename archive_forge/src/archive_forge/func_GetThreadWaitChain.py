from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetThreadWaitChain(WctHandle, Context=None, Flags=WCTP_GETINFO_ALL_FLAGS, ThreadId=-1, NodeCount=WCT_MAX_NODE_COUNT):
    _GetThreadWaitChain = windll.advapi32.GetThreadWaitChain
    _GetThreadWaitChain.argtypes = [HWCT, LPDWORD, DWORD, DWORD, LPDWORD, PWAITCHAIN_NODE_INFO, LPBOOL]
    _GetThreadWaitChain.restype = bool
    _GetThreadWaitChain.errcheck = RaiseIfZero
    dwNodeCount = DWORD(NodeCount)
    NodeInfoArray = (WAITCHAIN_NODE_INFO * NodeCount)()
    IsCycle = BOOL(0)
    _GetThreadWaitChain(WctHandle, Context, Flags, ThreadId, byref(dwNodeCount), ctypes.cast(ctypes.pointer(NodeInfoArray), PWAITCHAIN_NODE_INFO), byref(IsCycle))
    while dwNodeCount.value > NodeCount:
        NodeCount = dwNodeCount.value
        NodeInfoArray = (WAITCHAIN_NODE_INFO * NodeCount)()
        _GetThreadWaitChain(WctHandle, Context, Flags, ThreadId, byref(dwNodeCount), ctypes.cast(ctypes.pointer(NodeInfoArray), PWAITCHAIN_NODE_INFO), byref(IsCycle))
    return ([WaitChainNodeInfo(NodeInfoArray[index]) for index in compat.xrange(dwNodeCount.value)], bool(IsCycle.value))