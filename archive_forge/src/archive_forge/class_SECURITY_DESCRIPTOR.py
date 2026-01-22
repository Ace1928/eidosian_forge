import builtins
import ctypes.wintypes
from paramiko.util import u
class SECURITY_DESCRIPTOR(ctypes.Structure):
    """
    typedef struct _SECURITY_DESCRIPTOR
        {
        UCHAR Revision;
        UCHAR Sbz1;
        SECURITY_DESCRIPTOR_CONTROL Control;
        PSID Owner;
        PSID Group;
        PACL Sacl;
        PACL Dacl;
        }   SECURITY_DESCRIPTOR;
    """
    SECURITY_DESCRIPTOR_CONTROL = ctypes.wintypes.USHORT
    REVISION = 1
    _fields_ = [('Revision', ctypes.c_ubyte), ('Sbz1', ctypes.c_ubyte), ('Control', SECURITY_DESCRIPTOR_CONTROL), ('Owner', ctypes.c_void_p), ('Group', ctypes.c_void_p), ('Sacl', ctypes.c_void_p), ('Dacl', ctypes.c_void_p)]