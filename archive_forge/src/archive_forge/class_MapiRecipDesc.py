import os
from ctypes import *
class MapiRecipDesc(Structure):
    _fields_ = [('ulReserved', c_ulong), ('ulRecipClass', c_ulong), ('lpszName', c_char_p), ('lpszAddress', c_char_p), ('ulEIDSize', c_ulong), ('lpEntryID', c_void_p)]