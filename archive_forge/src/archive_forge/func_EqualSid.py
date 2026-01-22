from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def EqualSid(pSid1, pSid2):
    _EqualSid = windll.advapi32.EqualSid
    _EqualSid.argtypes = [PSID, PSID]
    _EqualSid.restype = bool
    return _EqualSid(pSid1, pSid2)