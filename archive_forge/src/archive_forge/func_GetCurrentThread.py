from winappdbg.win32.defines import *
def GetCurrentThread():
    _GetCurrentThread = windll.kernel32.GetCurrentThread
    _GetCurrentThread.argtypes = []
    _GetCurrentThread.restype = HANDLE
    return _GetCurrentThread()