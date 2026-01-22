from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def IsValidSid(pSid):
    _IsValidSid = windll.advapi32.IsValidSid
    _IsValidSid.argtypes = [PSID]
    _IsValidSid.restype = bool
    return _IsValidSid(pSid)