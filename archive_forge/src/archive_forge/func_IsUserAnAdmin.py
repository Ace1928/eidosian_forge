from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def IsUserAnAdmin():
    _IsUserAnAdmin = windll.shell32.IsUserAnAdmin
    _IsUserAnAdmin.argtypes = []
    _IsUserAnAdmin.restype = bool
    return _IsUserAnAdmin()