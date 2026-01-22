from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_PRIMARY_GROUP(Structure):
    _fields_ = [('PrimaryGroup', PSID)]