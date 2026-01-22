import ctypes
from os_win.utils.winapi import wintypes
class TRUSTEE(ctypes.Structure):
    _fields_ = [('pMultipleTrustee', wintypes.PVOID), ('MultipleTrusteeOperation', wintypes.INT), ('TrusteeForm', wintypes.INT), ('TrusteeType', wintypes.INT), ('pstrName', wintypes.LPWSTR)]