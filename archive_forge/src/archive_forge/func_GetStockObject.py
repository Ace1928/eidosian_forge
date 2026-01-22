from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
def GetStockObject(fnObject):
    _GetStockObject = windll.gdi32.GetStockObject
    _GetStockObject.argtypes = [ctypes.c_int]
    _GetStockObject.restype = HGDIOBJ
    _GetStockObject.errcheck = RaiseIfZero
    return _GetStockObject(fnObject)