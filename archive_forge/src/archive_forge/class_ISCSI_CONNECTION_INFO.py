import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_CONNECTION_INFO(ctypes.Structure):
    _fields_ = [('ConnectionId', ISCSI_UNIQUE_CONNECTION_ID), ('InitiatorAddress', wintypes.PWSTR), ('TargetAddress', wintypes.PWSTR), ('InitiatorSocket', wintypes.USHORT), ('TargetSocket', wintypes.USHORT), ('CID', ctypes.c_ubyte * 2)]