import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class SCSI_LUN_LIST(ctypes.Structure):
    _fields_ = [('OSLUN', wintypes.ULONG), ('TargetLUN', wintypes.ULONGLONG)]