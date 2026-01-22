from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class WAITCHAIN_NODE_INFO(Structure):
    _fields_ = [('ObjectType', WCT_OBJECT_TYPE), ('ObjectStatus', WCT_OBJECT_STATUS), ('u', _WAITCHAIN_NODE_INFO_UNION)]