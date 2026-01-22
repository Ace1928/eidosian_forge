from winappdbg.win32.defines import *
def GetSystemMetrics(nIndex):
    _GetSystemMetrics = windll.user32.GetSystemMetrics
    _GetSystemMetrics.argtypes = [ctypes.c_int]
    _GetSystemMetrics.restype = ctypes.c_int
    return _GetSystemMetrics(nIndex)